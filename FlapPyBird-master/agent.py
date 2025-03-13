import asyncio
import random
import pygame
import json
from src.flappy import Flappy
from src.entities import Background, Floor, Player, Pipes, Score, Coins, PlayerMode

class AutoPlayer:
    def __init__(self):
        self.base_flap_probability = 0.0015
        self.history = None
        

    def should_flap(self, player, pipes, coins, game):
        state = self.get_state(player, pipes, coins, game)

        if not state:
            return False

        # Calculate flap probability based on state
        prob = self.base_flap_probability

        # Increase probability if bird is falling too fast
        if state['vy'] > 5:
            prob += 0.1
        
        # Increase probability if bird is too low and falling
        if state['height'] > 300 and state['vy'] > 0:
            prob += 0.2
            
        # Increase probability if bird is approaching a pipe and needs to go up
        if state['px'] < 200 and state['py'] < -50:
            prob += 0.15
            
        # Increase probability if there's a coin above the bird
        if abs(state['cx']) < 200 and state['cy'] < -20:
            prob += 0.1

        # Cap probability at 0.95
        prob = min(prob, 0.95)
        
        return random.random() < prob

    def get_state(self, player, pipes, coins, score):
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
            distance = coin.x - player.x
            if distance < min_coin_distance:
                min_coin_distance = distance
                nearest_coin = coin

        if nearest_pipe or nearest_coin:
            return {
                'height': player.y,
                'px': nearest_pipe.x - player.x if nearest_pipe else 1000,
                'py': nearest_pipe.y - player.y + 380 if nearest_pipe else 0,
                'cx': nearest_coin.x - player.x if nearest_coin else 1000,
                'cy': nearest_coin.y - player.y if nearest_coin else 0,
                'vy': player.vel_y,
                'score': score.score,  # Add score to state
                'coins': score.coins_collected,
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
            current_state = player.get_state(game.player, game.pipes, game.coins, game.score)
            should_flap = player.should_flap(game.player, game.pipes, game.coins, game.score)
            if current_state:
                current_state['flapped'] = should_flap
                episode_states.append(current_state)

            if game.player.collided(game.pipes, game.floor):
                # Record terminal state
                episode_states.append({'state': 'dead'})
                player.history = episode_states
                break

                       # Auto flap with probability
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

            pygame.display.update()
            await asyncio.sleep(0)
            game.config.tick()

        # Save history after each episode
        player.save_history()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(auto_play())