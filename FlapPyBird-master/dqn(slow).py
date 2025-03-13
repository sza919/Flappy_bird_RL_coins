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
from src.flappy import Flappy
from src.entities import Background, Floor, Player, Pipes, Score, Coins, PlayerMode

# Neural network architecture for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No activation on output layer for unbounded Q-values
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.state_dim = 6  # [y_pos, y_vel, pipe_x, pipe_y, coin_x, coin_y]
        self.action_dim = 2  # [no flap, flap]
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.9999  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
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
        self.total_episodes = 0
        self.previous_state = None
        self.previous_action = None
        self.previous_score = 0  # Track previous score to calculate difference
        
    def normalize_state(self, state_dict):
        """Normalize the state values for neural network input"""
        # Extract values from state dictionary
        height = state_dict['height'] / 512.0  # Normalize by screen height
        vel_y = state_dict['vy'] / 10.0  # Normalize velocity
        pipe_x = min(state_dict['px'], 288) / 288.0  # Normalize by screen width
        pipe_y = state_dict['py'] / 512.0  # Normalize by screen height
        coin_x = min(state_dict['cx'], 288) / 288.0  # Normalize by screen width
        coin_y = state_dict['cy'] / 512.0  # Normalize by screen height
        
        # Return normalized values as a numpy array
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
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
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
            self.epsilon = checkpoint['epsilon']
            print(f"Loaded DQN model after {self.total_episodes} episodes")
            return True
        except (FileNotFoundError, ValueError):
            print("No valid DQN model found, starting fresh")
            return False

async def train_dqn_agent():
    game = Flappy()
    agent = DQNAgent()
    agent.load_model()  # Try to load existing model
    
    print("Starting DQN training...")
    
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
        agent.previous_score = 0  # Reset score tracking at episode start
        episode_steps = 0
        
        # Main game loop
        while True:
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
                if agent.previous_state is not None:
                    # Add terminal transition to memory with negative reward
                    agent.memory.add(
                        agent.previous_state,
                        agent.previous_action,
                        -10,  # Fixed negative reward for dying
                        agent.previous_state,  # Use previous state as terminal state doesn't matter
                        True  # Done flag
                    )
                break
                
            # If we have a valid state
            if current_state_dict:
                current_state = agent.normalize_state(current_state_dict)
                
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
                    reward += 0.1
                    
                    # Add small additional reward for good positioning
                    if agent.previous_state is not None:
                        pipe_y = agent.previous_state[3] * 512  # Denormalize pipe_y
                        pipe_x = agent.previous_state[2] * 288  # Denormalize pipe_x
                        
                        # Encourage bird to stay at middle level of the pipe
                        vertical_alignment = 0.2-abs(pipe_y)/512  # Higher when close to pipe center
                        
                        # Encourage progress toward pipe but with less weight
                        proximity_reward = -pipe_x/288 * 0.2
                        
                        reward += vertical_alignment + proximity_reward
                
                # Update previous score for next iteration
                agent.previous_score = current_score
                
                # Add to episode reward
                episode_reward += reward
                
                # Store transition in memory if we have both states
                if next_state_dict:
                    next_state = agent.normalize_state(next_state_dict)
                    
                    # Only add to memory if we have valid previous state
                    if agent.previous_state is not None:
                        agent.memory.add(
                            agent.previous_state,
                            agent.previous_action,
                            reward,
                            next_state,
                            False  # Not done yet
                        )
                
                # Learn from past experiences (batch learning)
                if len(agent.memory) > agent.batch_size:
                    agent.learn_from_experiences()
                
            pygame.display.update()
            await asyncio.sleep(0)
            game.config.tick()
            episode_steps += 1
        
        # Episode finished
        agent.total_episodes += 1
        agent.episode_rewards.append(episode_reward)
        
        # Update exploration rate
        agent.update_exploration_rate()
        
        # Update target network periodically
        if agent.total_episodes % agent.target_update == 0:
            agent.update_target_network()
            
        # Display progress and save model
        if agent.total_episodes % 10 == 0:
            avg_reward = sum(agent.episode_rewards[-10:]) / 10
            print(f"Episode {agent.total_episodes}, Avg Reward: {avg_reward:.2f}, Exploration: {agent.epsilon:.4f}, Memory: {len(agent.memory)}")
            agent.save_model()  # Save progress periodically
            
        await asyncio.sleep(0.5)  # Small delay between episodes

if __name__ == "__main__":
    asyncio.run(train_dqn_agent())