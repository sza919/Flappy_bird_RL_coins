import asyncio
import random
import pygame
from src.flappy import Flappy
from src.entities import Background, Floor, Player, Pipes, Score, Coins, PlayerMode

class AutoPlayer:
    def __init__(self):
        self.flap_probability = 0.04

    def should_flap(self):
        return random.random() < self.flap_probability

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
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if game.player.collided(game.pipes, game.floor):
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

        # Wait briefly before starting new game
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(auto_play())