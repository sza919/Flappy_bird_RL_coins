import asyncio
import random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
from src.flappy import Flappy
from src.entities import Background, Floor, Player, Pipes, Score, Coins, PlayerMode

# Neural network architecture for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # No activation on output layer for unbounded Q-values
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.best_trajectory = []
        self.best_reward = -float('inf')  # Initialize with negative infinity
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def add_trajectory(self, trajectory, reward):
        """Store a full trajectory if it has the highest reward seen so far"""
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_trajectory = trajectory.copy()
            #print(f"New best trajectory saved with reward: {reward}")
    
    def sample(self, batch_size):
        # If we don't have enough samples in the buffer, return what we have
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        # Include experiences from best trajectory if available
        if self.best_trajectory:
            # Calculate how many samples to take from regular buffer vs best trajectory
            best_samples = min(batch_size // 4, len(self.best_trajectory))
            regular_samples = min(batch_size - best_samples, len(self.buffer))
            
            # Sample from regular buffer
            regular_experiences = random.sample(list(self.buffer), regular_samples) if regular_samples > 0 else []
            
            # Sample from best trajectory
            best_experiences = random.sample(self.best_trajectory, best_samples) if best_samples > 0 else []
            
            # Combine samples
            return regular_experiences + best_experiences
        else:
            return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.state_dim = 3  # [vy, px, py]
        self.action_dim = 2  # [no flap, flap]
        
        # Hyperparameters
        self.learning_rate = 0.001
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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer(10000)
        
        # Training stats
        self.episode_rewards = []
        self.episode_scores = []  # Track scores for each episode
        self.total_episodes = 0
        self.previous_state = None
        self.previous_action = None
        self.previous_score = 0  # Track previous score to calculate difference
        
    def normalize_state(self, state_dict):
        """Normalize the state values for neural network input"""
        # Extract values from state dictionary
        
        vel_y = state_dict['vy'] / 10.0  # Normalize velocity
        pipe_x = min(state_dict['px'], 288) / 288.0  # Normalize by screen width
        pipe_y = state_dict['py'] / 512.0  # Normalize by screen height
        
        # Return normalized values as a numpy array
        return np.array([vel_y, pipe_x, pipe_y], dtype=np.float32)
    
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
        # Check if we have enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        transitions = self.memory.sample(self.batch_size)
        
        # Convert batch of transitions to separate arrays
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert to numpy arrays first, then to tensors (much faster)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute next Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Compute target Q values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_exploration_rate(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self.total_episodes,
            'rewards': self.episode_rewards,
            'scores': self.episode_scores,  # Also save scores history
            'epsilon': self.epsilon
        }, 'dqn_model.pth')
        
    def load_model(self):
        try:
            checkpoint = torch.load('dqn_model.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.total_episodes = checkpoint['episodes']
            self.episode_rewards = checkpoint['rewards']
            self.episode_scores = checkpoint.get('scores', [])  # Handle case where old model doesn't have scores
            self.epsilon = checkpoint['epsilon']
            print(f"Loaded DQN model after {self.total_episodes} episodes")
            return True
        except (FileNotFoundError, ValueError):
            print("No valid DQN model found, starting fresh")
            return False

# Add this function to GameConfig class by monkey patching
def tick_no_delay(self):
    """Tick without enforcing FPS limit"""
    self.clock.tick()  # Just update the clock without delay

async def train_dqn_agent(display=False):
    # Initialize game without display
    if not display:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Set dummy video driver
    
    game = Flappy(headless=not display)  # Assuming Flappy class accepts a headless parameter
    agent = DQNAgent()
    agent.display = display
    agent.load_model()  # Try to load existing model
    
    print("Starting DQN training in headless mode...")
    
    # Disable FPS limit to maximize training speed
    if not display:
        game.config.fps = 0  # Set to zero to remove FPS cap
        game.config.tick_no_delay = tick_no_delay.__get__(game.config, type(game.config))
    
    while True:
        # Initialize game components
        game.background = Background(game.config)
        game.floor = Floor(game.config)
        game.player = Player(game.config)
        game.pipes = Pipes(game.config)
        game.score = Score(game.config)
        game.coins = Coins(game.config)
        game.config.coins = game.coins
        
        # Skip splash screen
        game.player.set_mode(PlayerMode.NORMAL)
        
        # Variables for this episode
        episode_reward = 0
        agent.previous_state = None
        agent.previous_action = None
        agent.previous_score = 0
        episode_steps = 0
        episode_trajectory = []  # Store all transitions for this episode
        
        # Main game loop
        while True:
            # Check for quit events but skip rendering-related events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save_model()
                    pygame.quit()
                    return
            
            # Get current state
            current_state_dict = agent.get_state(game.player, game.pipes, game.coins, game.score)
            
            # Check if game is over
            game_over = game.player.collided(game.pipes, game.floor)
            
            if game_over:
                terminal_reward = -10
                # Add small additional reward for good positioning
                if agent.previous_state is not None:
                    pipe_y = agent.previous_state[1] * 512  # Denormalize pipe_y
                    # Encourage bird to stay at middle level of the pipe
                    vertical_alignment = -abs(pipe_y)/512  # Higher when close to pipe center
                    
                    terminal_reward += 10*vertical_alignment 
                if agent.previous_state is not None and current_state_dict:
                    # Add terminal transition to memory with negative reward
                    current_state = agent.normalize_state(current_state_dict)
                    terminal_transition = (
                        agent.previous_state,
                        agent.previous_action,
                        terminal_reward,  # Fixed negative reward for dying
                        current_state,  # Use current state as the terminal state
                        True  # Done flag
                    )
                    agent.memory.add(*terminal_transition)
                    episode_trajectory.append(terminal_transition)
                    episode_reward += terminal_reward
                break
                
            # If we have a valid state
            if current_state_dict:
                # Choose and perform action
                should_flap = agent.should_flap(game.player, game.pipes, game.coins, game.score)
                if should_flap:
                    game.player.flap()
                
                # Check coins
                for coin in game.coins.coins[:]:
                    if game.player.collide(coin):
                        game.score.add_coins(1)
                        game.coins.coins.remove(coin)
                
                # Check score
                for pipe in game.pipes.upper:
                    if game.player.crossed(pipe):
                        game.score.add()
                
                # Update game state
                game.background.tick()
                game.floor.tick()
                game.pipes.tick()
                game.score.tick()
                game.coins.tick(game.pipes)
                game.player.tick()
                
                # Get next state
                next_state_dict = agent.get_state(game.player, game.pipes, game.coins, game.score)
                
                # Calculate reward
                current_score = game.score.score
                reward = current_score - agent.previous_score
                
                # Small reward for staying alive
                if reward == 0:
                    # Add small additional reward for good positioning
                    reward += 0.03
                    if should_flap:
                        reward -= 0.3
                        # can be removed
                        
                # Update previous score for next iteration
                agent.previous_score = current_score
                
                # Add to episode reward
                episode_reward += reward
                
                # Store transition in memory if we have both states
                if next_state_dict:
                    next_state = agent.normalize_state(next_state_dict)
                    
                    # Only add to memory if we have valid previous state
                    if agent.previous_state is not None:
                        transition = (
                            agent.previous_state,
                            agent.previous_action,
                            reward,
                            next_state,
                            False  # Not done yet
                        )
                        agent.memory.add(*transition)
                        episode_trajectory.append(transition)
                
                # Learn from past experiences (batch learning)
                if len(agent.memory) > agent.batch_size:
                    agent.learn_from_experiences()
                
            # Skip display update and use no delay for tick
            if not display:
                game.config.tick_no_delay()
            else:
                pygame.display.update()
                await asyncio.sleep(0)
                game.config.tick()
            episode_steps += 1
            
            # No sleep to maximize training speed - completely removed
        
        # Episode finished
        # Store this episode trajectory if it's the best so far
        agent.memory.add_trajectory(episode_trajectory, episode_reward)
        
        agent.total_episodes += 1
        agent.episode_rewards.append(episode_reward)
        agent.episode_scores.append(game.score.score)  # Store final score
        
        # Update exploration rate
        agent.update_exploration_rate()
        
        # Update target network periodically
        if agent.total_episodes % agent.target_update == 0:
            agent.update_target_network()
            
        # Display progress and save model
        if agent.total_episodes % 10 == 0:
            avg_reward = sum(agent.episode_rewards[-10:]) / 10
            
            # Display scores from last 10 episodes
            recent_scores = agent.episode_scores[-10:]
            scores_str = ", ".join([f"{score}" for score in recent_scores])
            print(f"Episode {agent.total_episodes}, Avg Reward: {avg_reward:.2f}")
            print(f"Last 10 rewards: {[round(r,2) for r in agent.episode_rewards[-10:]]}")
            print(f"Last 10 scores: [{scores_str}], Avg Score: {sum(recent_scores)/len(recent_scores):.1f}")
            print(f"Memory size: {len(agent.memory)}")
            print(f"Best reward so far: {agent.memory.best_reward}")
            
            agent.save_model()  # Save progress periodically
        
        # Minimal delay between episodes (but still allow some breathing room for system)
        if not display:
            await asyncio.sleep(0.001)
        else:
            await asyncio.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Flappy Bird with DQN')
    parser.add_argument('--display', action='store_true', help='Enable display mode')
    args = parser.parse_args()
    
    asyncio.run(train_dqn_agent(display=args.display))