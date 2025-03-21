import asyncio
import random
import pygame
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import os
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

class DQNAgent:
    def __init__(self):
        self.state_dim = 6  # [y_pos, y_vel, pipe_x, pipe_y, coin_x, coin_y]
        self.action_dim = 2  # [no flap, flap]
        
        # Neural Networks
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        
        # Training stats
        self.total_episodes = 0
        self.episode_scores = []
        self.episode_coins = []
        
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
        # Always use the policy network for action selection (no exploration)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  # Ensure no gradients are computed
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def should_flap(self, player, pipes, coins, game):
        state_dict = self.get_state(player, pipes, coins, game)
        
        if not state_dict:
            return False
            
        state = self.normalize_state(state_dict)
        action = self.choose_action(state)
        
        return action == 1  # Return True if action is flap (1)
    
    def load_model(self):
        """Load the trained model"""
        try:
            checkpoint = torch.load('dqn_model.pth', weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            # Set policy network to evaluation mode
            self.policy_net.eval()
            print("Model loaded successfully!")
            return True
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print("No valid DQN model found")
            return False

def tick_no_delay(self):
    """Tick without enforcing FPS limit"""
    self.clock.tick()

async def play_dqn_agent(display=True, num_episodes=10, name="play"):
    if not display:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    game = Flappy(headless=not display)
    agent = DQNAgent()
    
    if not agent.load_model():
        print("Failed to load model. Exiting...")
        return
    
    print("Starting DQN play mode...")
    print("Press Ctrl+C to exit...")
    
    # Initialize play metrics to match training metrics format
    play_metrics = {
        'episodes': 0,
        'scores': [],
        'coins_collected': [],
        'max_scores': [],
        'rolling_mean_scores': [],
        'max_score': 0,
        'total_coins': 0
    }
    
    # Create filename with name parameter
    metrics_filename = f'play_metrics_{name}.json'
    
    if not display:
        game.config.fps = 0
        game.config.tick_no_delay = tick_no_delay.__get__(game.config, type(game.config))
    
    try:
        while agent.total_episodes < num_episodes:
            game.background = Background(game.config)
            game.floor = Floor(game.config)
            game.player = Player(game.config)
            game.pipes = Pipes(game.config)
            game.score = Score(game.config)
            game.coins = Coins(game.config)
            game.config.coins = game.coins
            
            game.player.set_mode(PlayerMode.NORMAL)
            frame_count = 0
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    
                frame_count += 1
                current_state_dict = agent.get_state(game.player, game.pipes, game.coins, game.score)
                game_over = game.player.collided(game.pipes, game.floor)
                
                if game_over or frame_count > 900:
                    # Save episode metrics
                    play_metrics['episodes'] += 1
                    play_metrics['scores'].append(game.score.score)
                    play_metrics['coins_collected'].append(game.score.coins_collected)
                    
                    # Update max score and max_scores list
                    play_metrics['max_score'] = max(play_metrics['max_score'], game.score.score)
                    play_metrics['max_scores'].append(play_metrics['max_score'])
                    
                    # Calculate rolling mean score
                    window_size = min(100, len(play_metrics['scores']))
                    if window_size > 0:
                        rolling_mean = sum(play_metrics['scores'][-window_size:]) / window_size
                        play_metrics['rolling_mean_scores'].append(rolling_mean)
                    
                    play_metrics['total_coins'] += game.score.coins_collected
                    
                    # Save metrics after each episode
                    with open(metrics_filename, 'w') as f:
                        json.dump(play_metrics, f, indent=4)
                    
                    print(f"\nEpisode {play_metrics['episodes']} completed:")
                    print(f"Score: {game.score.score}")
                    print(f"Coins collected: {game.score.coins_collected}")
                    print(f"Best score so far: {play_metrics['max_score']}")
                    print(f"Total coins collected: {play_metrics['total_coins']}")
                    break
                    
                if current_state_dict:
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
                
                if not display:
                    game.config.tick_no_delay()
                else:
                    pygame.display.update()
                    await asyncio.sleep(0)
                    game.config.tick()
            
            agent.total_episodes += 1
            
            if not display:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(1)
        
        print("\nPlay session completed!")
        print(f"Total episodes: {play_metrics['episodes']}")
        print(f"Best score: {play_metrics['max_score']}")
        print(f"Total coins collected: {play_metrics['total_coins']}")
        print(f"Average score: {sum(play_metrics['scores'])/len(play_metrics['scores']):.2f}")
        print(f"Average coins per episode: {play_metrics['total_coins']/play_metrics['episodes']:.2f}")

    except KeyboardInterrupt:
        print("\nPlay session interrupted.")
        print(f"Total episodes completed: {play_metrics['episodes']}")
        print(f"Best score: {play_metrics['max_score']}")
        print(f"Total coins collected: {play_metrics['total_coins']}")
        pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Play Flappy Bird with trained DQN model')
    parser.add_argument('--display', action='store_true', help='Enable display mode')
    parser.add_argument('--episodes', type=int, default=300, help='Number of episodes to play')
    parser.add_argument('--name', type=str, default='play', help='Name for the play session data file')
    args = parser.parse_args()
    
    asyncio.run(play_dqn_agent(display=args.display, num_episodes=args.episodes, name=args.name)) 