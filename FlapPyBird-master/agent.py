import asyncio
import random
import pygame
import json
from src.flappy import Flappy
from src.entities import Background, Floor, Player, Pipes, Score, Coins, PlayerMode

class AutoPlayer:
    def __init__(self):
        self.flap_probability = 0.04
        self.history = []

    def should_flap(self):
        return random.random() < self.flap_probability

    def get_state(self, player, pipes, coins):
        # Get nearest pipe
        nearest_pipe = None
        min_distance = float('inf')
        for pipe in pipes.upper:
            distance = pipe.x - player.x
            if distance < min_distance:
                min_distance = distance
                nearest_pipe = pipe

        # Get nearest coin
        nearest_coin = None
        min_coin_distance = float('inf')
        for coin in coins.coins:
            distance = ((coin.x - player.x)**2 + (coin.y - player.y)**2)**0.5
            if distance < min_coin_distance:
                min_coin_distance = distance
                nearest_coin = coin

        if nearest_pipe or nearest_coin:
            return {
                'px': nearest_pipe.x - player.x if nearest_pipe else 1000,
                'py': nearest_pipe.y - player.y if nearest_pipe else 0,
                'cx': nearest_coin.x - player.x if nearest_coin else 1000,
                'cy': nearest_coin.y - player.y if nearest_coin else 0,
                'vy': player.vel_y,
                'state': 'alive'
            }
        return None

    def save_history(self):
        with open('play_history.json', 'w') as f:
            json.dump(self.history, f)

async def auto_play():
    game = Flappy()
    player = AutoPlayer()
    
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
        
        # Main game loop
        episode_states = []
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    player.save_history()
                    pygame.quit()
                    return

            # Record current state
            current_state = player.get_state(game.player, game.pipes, game.coins)
            if current_state:
                episode_states.append(current_state)

            if game.player.collided(game.pipes, game.floor):
                # Record terminal state
                episode_states.append({'state': 'dead'})
                player.history.append(episode_states)
                break

                       # Auto flap with probability
            if player.should_flap():
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

            pygame.display.update()
            await asyncio.sleep(0)
            game.config.tick()

        # Save history after each episode
        player.save_history()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(auto_play())