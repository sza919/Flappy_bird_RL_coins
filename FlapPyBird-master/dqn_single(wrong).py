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
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

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
        self.learning_rate = 0.05
        self.gamma = 1  # Discount factor
        self.epsilon = 0.03 # Exploration rate
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.00001
        self.batch_size = 64
        self.target_update = 10  # Update target network every N episodes
        
        # Neural Networks
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
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
    
    def learn_online(self,state,action,reward,next_state,done):
        # Check if we have enough samples in memory
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.LongTensor([action]).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)
        done_tensor = torch.FloatTensor([done]).unsqueeze(0)

        # Compute current Q value
        current_q = self.policy_net(state_tensor).gather(1, action_tensor)

        # Compute next Q value (using target network)
        next_q = self.target_net(next_state_tensor).max(1)[0].unsqueeze(1).detach()
        expected_q = reward_tensor + self.gamma * next_q * (1 - done_tensor)

        # Compute loss
        loss = F.smooth_l1_loss(current_q, expected_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (optional)
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
            
            # if game_over:
            #     if agent.previous_state is not None:
            #         # Add terminal transition to memory
            #         agent.memory.add(
            #             agent.previous_state, 
            #             agent.previous_action, 
            #             -100,  # Big negative reward for dying
            #             agent.previous_state,  # Use previous state as terminal state doesn't matter
            #             True  # Done flag
            #         )
            #     break
            if game_over:
                if agent.previous_state is not None:
                    # Calculate distance to the nearest pipe
                    # Extract pipe distance and height from previous state
                    pipe_y = agent.previous_state[3] * 512.0  # Denormalize pipe_y
                    # this is distance!!!
                    # Add a reward if the bird was close to passing the pipe
                
                    close_to_passing_loss = abs(pipe_y)

                    # Add terminal transition to memory
                    agent.memory.add(
                        agent.previous_state,
                        agent.previous_action,
                        -100*close_to_passing_loss,  # Adjusted reward
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
                
                # Calculate reward based on score difference
                current_score = game.score.score
                reward = current_score - agent.previous_score
                
                # Add small positive reward for staying alive
                if reward == 0:
                    pipe_y = agent.previous_state[3]*512  #normalized_distance
                    pipe_x = agent.previous_state[2]
                    height = agent.previous_state[0]*512 #unnormalized height
                    
                    # Add a reward if the bird was close to passing the pipe
                
                    vertical_distance = abs(pipe_y)
                    # Reward is inversely proportional to the vertical distance
                    close_to_passing_loss = vertical_distance
                    
                    reward +=   -abs(pipe_y)/512  - pipe_x # More reward the closer it is
                    
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
                
                # Learn from past experiences
                agent.learn_online(current_state,should_flap,reward,next_state,False)
                
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