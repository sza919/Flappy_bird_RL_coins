import asyncio
import random
import pygame
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import pandas as pd
from src.flappy import Flappy
from src.entities import Background, Floor, Player, Pipes, Score, Coins, PlayerMode

os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Neural network architecture for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.best_trajectory = []
        self.best_reward = -float('inf')
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def add_trajectory(self, trajectory, total_reward):
        """Store a full trajectory if it has the highest reward seen so far"""
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_trajectory = trajectory.copy()
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        # Include experiences from best trajectory if available
        if self.best_trajectory:
            # Calculate how many samples to take from regular buffer vs best trajectory
            best_samples = min(batch_size // 4, len(self.best_trajectory))
            regular_samples = batch_size - best_samples
            
            # Sample from regular buffer
            regular_experiences = random.sample(list(self.buffer), regular_samples)
            
            # Sample from best trajectory
            best_experiences = random.sample(self.best_trajectory, best_samples)
            
            # Combine samples
            return regular_experiences + best_experiences
        else:
            return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.state_dim = 6  # [y_pos, y_vel, pipe_x, pipe_y, coin_x, coin_y]
        self.action_dim = 2  # [no flap, flap]
        
        # Hyperparameters
        self.learning_rate = 0.0001
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0
        self.batch_size = 256
        self.target_update = 10  # Update target network every N episodes
        
        # Neural Networks
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=0.001)
        
        # Replay memory
        self.memory = ReplayBuffer(100000)
        
        # Training stats
        self.episode_rewards = []
        self.episode_scores = []  # Track scores for each episode
        self.total_episodes = 0
        self.previous_state = None
        self.previous_action = None
        self.previous_score = 0  # Track previous score to calculate difference
        
        # Analysis metrics
        self.max_scores = []
        self.rolling_mean_scores = []
        
    def normalize_state(self, state_dict):
        """Normalize the state values for neural network input"""
        height = state_dict['height'] / 512.0  # Normalize by screen height
        vel_y = state_dict['vy'] / 10.0  # Normalize velocity
        pipe_x = min(state_dict['px'], 288) / 288.0  # Normalize by screen width
        pipe_y = state_dict['py'] / 512.0  # Normalize by screen height
        coin_x = min(state_dict['cx'], 288) / 288.0  # Normalize by screen width
        coin_y = state_dict['cy'] / 512.0  # Normalize by screen height
        
        return np.array([height, vel_y, pipe_x, pipe_y, coin_x, coin_y], dtype=np.float32)
    
    def get_state(self, player, pipes, coins, score):
        # Get nearest pipe
        nearest_pipe = None
        min_distance = float('inf')
        for pipe in pipes.upper:
            distance = pipe.x - player.x
            if distance >= 0 and distance < min_distance:  # Only consider pipes ahead of player
                min_distance = distance
                nearest_pipe = pipe
        
        # If no pipes ahead, find the first one
        if not nearest_pipe and pipes.upper:
            nearest_pipe = min(pipes.upper, key=lambda p: p.x if p.x > player.x else float('inf'))
        
        # Get nearest coin
        nearest_coin = None
        min_coin_distance = float('inf')
        for coin in coins.coins:
            distance = coin.x - player.x
            if distance >= 0 and distance < min_coin_distance:
                min_coin_distance = distance
                nearest_coin = coin

        if nearest_pipe:
            return {
                'height': player.y,
                'px': nearest_pipe.x - player.x if nearest_pipe else 1000,
                'py': nearest_pipe.y - player.y + 380 if nearest_pipe else 0,
                'cx': nearest_coin.x - player.x if nearest_coin else 1000,
                'cy': nearest_coin.y - player.y if nearest_coin else 0,
                'vy': player.vel_y,
                'score': score.score,
                'coins': score.coins_collected,
                'state': 'alive'
            }
        return None
    
    def choose_action(self, state):
        # Exploration (random action)
        if random.random() < self.epsilon:
            return random.choice([0, 1])  # 0: do nothing, 1: flap
            
        # Exploitation (best action from neural network)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def should_flap(self, player, pipes, coins, game):
        state_dict = self.get_state(player, pipes, coins, game)
        
        if not state_dict:
            return False
            
        state = self.normalize_state(state_dict)
        action = self.choose_action(state)
        self.previous_state = state
        self.previous_action = action
        
        return action == 1  # Return True if action is flap (1)
    
    def learn_from_experiences(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_exploration_rate(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self):

        memory_list = []
        for state, action, reward, next_state, done in self.memory.buffer:
            memory_list.append((
                state.tolist() if isinstance(state, np.ndarray) else float(state),
                int(action),
                float(reward),
                next_state.tolist() if isinstance(next_state, np.ndarray) else float(next_state),
                bool(done)
            ))

        best_trajectory = []
        for state, action, reward, next_state, done in self.memory.best_trajectory:
            memory_list.append((
                state.tolist() if isinstance(state, np.ndarray) else float(state),
                int(action),
                float(reward),
                next_state.tolist() if isinstance(next_state, np.ndarray) else float(next_state),
                bool(done)
            ))

        model_state = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self.total_episodes,
            'rewards': [float(r) for r in self.episode_rewards],
            'scores': self.episode_scores,
            'epsilon': float(self.epsilon),
            'max_scores': self.max_scores,
            'rolling_mean_scores': self.rolling_mean_scores,
            'best_trajectory': best_trajectory,  # Save best trajectory
            'best_reward': float(self.memory.best_reward)  # Save best reward
        }
        
        torch.save(model_state, 'dqn_model_best_tj.pth')
        
    def load_model(self):
        try:
            checkpoint = torch.load('dqn_model_best_tj.pth', weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.total_episodes = checkpoint['episodes']
            self.episode_rewards = checkpoint['rewards']
            self.episode_scores = checkpoint.get('scores', [])
            self.epsilon = checkpoint['epsilon']
            self.max_scores = checkpoint.get('max_scores', [])
            self.rolling_mean_scores = checkpoint.get('rolling_mean_scores', [])
            # Load best trajectory if available
            self.memory = ReplayBuffer(100000)
            if 'memory' in checkpoint:
                for state, action, reward, next_state, done in checkpoint['memory']:
                    self.memory.add(
                        np.array(state, dtype=np.float32),
                        action,
                        reward,
                        np.array(next_state, dtype=np.float32),
                        done
                    )
            
            # Restore best trajectory
            if 'best_trajectory' in checkpoint and checkpoint['best_trajectory']:
                self.memory.best_trajectory = []
                for exp in checkpoint['best_trajectory']:
                    state = np.array(exp['state'], dtype=np.float32)
                    action = exp['action']
                    reward = exp['reward']
                    next_state = np.array(exp['next_state'], dtype=np.float32)
                    done = exp['done']
                    self.memory.best_trajectory.append((state, action, reward, next_state, done))
                self.memory.best_reward = checkpoint['best_reward']
                
            print(f"Loaded DQN model after {self.total_episodes} episodes")
            return True
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print("No valid DQN model found, starting fresh")
            return False

def tick_no_delay(self):
    """Tick without enforcing FPS limit"""
    self.clock.tick()

async def train_dqn_agent(display=False, max_episodes=None):
    if not display:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    game = Flappy(headless=not display)
    agent = DQNAgent()
    agent.display = display
    agent.load_model()
    
    print("Starting DQN training...")
    print("Press Ctrl+C to save and exit...")
    if max_episodes:
        print(f"Training will stop after {max_episodes} episodes")
    
    # Initialize training metrics
    training_metrics = {
        'episodes': 0,
        'scores': [],
        'coins_collected': [],
        'max_scores': [],
        'rolling_mean_scores': []
    }
    
    try:
        with open('training_metrics_best_tj.json', 'r') as f:
            loaded_metrics = json.load(f)
            training_metrics['episodes'] = loaded_metrics.get('episodes', 0)
            training_metrics['scores'] = loaded_metrics.get('scores', [])
            training_metrics['coins_collected'] = loaded_metrics.get('coins_collected', [])
            training_metrics['max_scores'] = loaded_metrics.get('max_scores', [])
            training_metrics['rolling_mean_scores'] = loaded_metrics.get('rolling_mean_scores', [])
    except FileNotFoundError:
        pass
    
    if not display:
        game.config.fps = 0
        game.config.tick_no_delay = tick_no_delay.__get__(game.config, type(game.config))
    
    try:
        while True:
            if max_episodes and agent.total_episodes >= max_episodes:
                print(f"\nReached maximum episodes ({max_episodes}). Saving model and data...")
                agent.save_model()
                training_data = {
                    "episodes": agent.total_episodes,
                    "scores": training_metrics['scores'],
                    "coins_collected": training_metrics['coins_collected'],
                    "max_scores": agent.max_scores,
                    "rolling_mean_scores": agent.rolling_mean_scores
                }
                with open("training_metrics_best_tj.json", "w") as f:
                    json.dump(training_data, f, indent=4)
                
                print("Model and data saved successfully!")
                pygame.quit()
                return
            
            game.background = Background(game.config)
            game.floor = Floor(game.config)
            game.player = Player(game.config)
            game.pipes = Pipes(game.config)
            game.score = Score(game.config)
            game.coins = Coins(game.config)
            game.config.coins = game.coins
            
            game.player.set_mode(PlayerMode.NORMAL)
            
            episode_reward = 0
            agent.previous_state = None
            agent.previous_action = None
            agent.previous_score = 0
            episode_steps = 0
            episode_trajectory = []  # Initialize trajectory collection
            
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save_model()
                        training_metrics['episodes'] += 1
                        training_metrics['scores'].append(game.score.score)
                        training_metrics['coins_collected'].append(game.score.coins_collected)
                        with open('training_metrics_best_tj.json', 'w') as f:
                            json.dump(training_metrics, f, indent=4)
                        pygame.quit()
                        return
                
                current_state_dict = agent.get_state(game.player, game.pipes, game.coins, game.score)
                game_over = game.player.collided(game.pipes, game.floor)
                
                if game_over:
                    if agent.previous_state is not None:
                        terminal_transition = (
                            agent.previous_state,
                            agent.previous_action,
                            -10,
                            agent.previous_state,
                            True
                        )
                        agent.memory.add(*terminal_transition)
                        episode_trajectory.append(terminal_transition)
                        
                    # Add the complete trajectory to replay buffer
                    agent.memory.add_trajectory(episode_trajectory, episode_reward)
                    
                    training_metrics['episodes'] += 1
                    training_metrics['scores'].append(game.score.score)
                    training_metrics['coins_collected'].append(game.score.coins_collected)
                    
                    if training_metrics['episodes'] % 10 == 0:
                        with open('training_metrics_best_tj.json', 'w') as f:
                            json.dump(training_metrics, f, indent=4)
                        
                        if display:
                            print(f"Episode {training_metrics['episodes']}")
                            print(f"Score: {game.score.score}")
                            print(f"Coins: {game.score.coins_collected}")
                            print("-------------------")
                    break
                    
                if current_state_dict:
                    current_state = agent.normalize_state(current_state_dict)
                    
                    should_flap = agent.should_flap(game.player, game.pipes, game.coins, game.score)
                    if should_flap:
                        game.player.flap()
                    
                    for coin in game.coins.coins[:]:
                        if game.player.collide(coin):
                            game.score.add_coins(1)
                            game.coins.coins.remove(coin)
                    
                    for pipe in game.pipes.upper:
                        if game.player.crossed(pipe):
                            game.score.add()
                    
                    game.background.tick()
                    game.floor.tick()
                    game.pipes.tick()
                    game.score.tick()
                    game.coins.tick(game.pipes)
                    game.player.tick()
                    
                    next_state_dict = agent.get_state(game.player, game.pipes, game.coins, game.score)
                    
                    current_score = game.score.score
                    reward = current_score - agent.previous_score
                    current_state = agent.normalize_state(current_state_dict)
                    normalized_py = current_state[2]
                    normalized_cy = current_state[4]
                    normalized_cx = current_state[5]
                    if reward == 0:
                        reward += 0.03
                        v_p_dist = abs(normalized_py) * 512
                        v_c_dist = abs(normalized_cy) * 512
                        h_c_dist = abs(normalized_cx) * 288
                        if v_p_dist < 150:
                            reward += 0.002
                        if v_c_dist < 60:
                            reward += 0.001
                        if h_c_dist < 60:
                            reward += 0.001
                        if should_flap:
                            reward -= 0.3

                    agent.previous_score = current_score
                    episode_reward += reward
                    
                    if next_state_dict:
                        next_state = agent.normalize_state(next_state_dict)
                        
                        if agent.previous_state is not None:
                            transition = (
                                agent.previous_state,
                                agent.previous_action,
                                reward,
                                next_state,
                                False
                            )
                            agent.memory.add(*transition)
                            episode_trajectory.append(transition)
                    
                    if len(agent.memory) > agent.batch_size:
                        agent.learn_from_experiences()
                    
                if not display:
                    game.config.tick_no_delay()
                else:
                    pygame.display.update()
                    await asyncio.sleep(0)
                    game.config.tick()
                episode_steps += 1
            
            agent.total_episodes += 1
            agent.episode_rewards.append(episode_reward)
            agent.episode_scores.append(game.score.score)
            
            if not agent.max_scores or game.score.score > agent.max_scores[-1]:
                agent.max_scores.append(game.score.score)
            else:
                agent.max_scores.append(agent.max_scores[-1])
            
            window_size = 100
            window = agent.episode_scores[-window_size:] if len(agent.episode_scores) >= window_size else agent.episode_scores
            rolling_mean = sum(window) / len(window) if window else 0
            agent.rolling_mean_scores.append(rolling_mean)
            
            coins_window = training_metrics['coins_collected'][-window_size:] if len(training_metrics['coins_collected']) >= window_size else training_metrics['coins_collected']
            coins_rolling_mean = sum(coins_window) / len(coins_window) if coins_window else 0
            training_metrics['rolling_mean_coins'] = training_metrics.get('rolling_mean_coins', [])
            training_metrics['rolling_mean_coins'].append(coins_rolling_mean)
            
            if agent.total_episodes % 10 == 0:
                training_data = {
                    "episodes": agent.total_episodes,
                    "scores": training_metrics['scores'],
                    "coins_collected": training_metrics['coins_collected'],
                    "max_scores": agent.max_scores,
                    "rolling_mean_scores": agent.rolling_mean_scores,
                    "rolling_mean_coins": training_metrics['rolling_mean_coins']
                }
                with open("training_metrics_best_tj.json", "w") as f:
                    json.dump(training_data, f, indent=4)
            
            agent.update_exploration_rate()
            
            if agent.total_episodes % agent.target_update == 0:
                agent.update_target_network()
                
            if agent.total_episodes % 10 == 0:
                avg_reward = sum(agent.episode_rewards[-10:]) / 10
                recent_scores = agent.episode_scores[-10:]
                scores_str = ", ".join([f"{score}" for score in recent_scores])
                print(f"Episode {agent.total_episodes}, Avg Reward: {avg_reward:.2f}, Exploration: {agent.epsilon:.4f}")
                print(f"Last 10 scores: [{scores_str}], Avg Score: {sum(recent_scores)/len(recent_scores):.1f}")
                print(f"Coins collected this episode: {training_metrics['coins_collected'][-1] if training_metrics['coins_collected'] else 0}")
                print(f"Rolling avg coins (window=100): {coins_rolling_mean:.2f}")
                print(f"Memory size: {len(agent.memory)}\n")
                agent.save_model()
            
            if not display:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model and data...")
        agent.save_model()
        
        training_data = {
            "episodes": agent.total_episodes,
            "scores": training_metrics['scores'],
            "coins_collected": training_metrics['coins_collected'],
            "max_scores": agent.max_scores,
            "rolling_mean_scores": agent.rolling_mean_scores
        }
        with open("training_metrics_best_tj.json", "w") as f:
            json.dump(training_data, f, indent=4)
            
        print("Model and data saved successfully!")
        pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Flappy Bird with DQN and Analysis')
    parser.add_argument('--display', action='store_true', help='Enable display mode')
    parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    args = parser.parse_args()
    
    asyncio.run(train_dqn_agent(display=args.display, max_episodes=args.episodes))