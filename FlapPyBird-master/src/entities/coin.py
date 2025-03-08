import random

import pygame

from ..utils import GameConfig
from .entity import Entity


class Coin(Entity):
    def __init__(self, config: GameConfig, x: int, y: int) -> None:
        super().__init__(
            config=config,
            image=config.images.coin,
            x=x,
            y=y,
        )
        self.vel_x = -5  # Same speed as pipes

    def draw(self) -> None:
        """Update position and draw the coin"""
        self.x += self.vel_x
        super().draw()


class Coins(Entity):
    def __init__(self, config: GameConfig) -> None:
        super().__init__(config)
        self.coins = []
        self.spawn_timer = 0
        self.spawn_interval = 90  # Spawn new coins every 90 frames
        self.height_range = (150, 350)  # Fixed vertical range for coins

    def add_coin(self, x: int, y: int) -> None:
        self.coins.append(Coin(self.config, x, y))

    def is_position_safe(self, x: int, y: int, pipes) -> bool:
        """Check if a position is safe (not overlapping with pipes)"""
        # Create a temporary rect for the coin
        coin_rect = pygame.Rect(
            x,
            y,
            self.config.images.coin.get_width(),
            self.config.images.coin.get_height(),
        )

        # Check against all pipes
        for pipe in pipes.upper + pipes.lower:
            if coin_rect.colliderect(pipe.rect):
                return False
        return True

    def find_safe_position(self, pipes) -> tuple[int, int]:
        """Find a safe position for a new coin"""
        x = self.config.window.width
        attempts = 10  # Maximum attempts to find a safe position

        while attempts > 0:
            y = random.randint(*self.height_range)
            if self.is_position_safe(x, y, pipes):
                return x, y
            attempts -= 1

        return None  # No safe position found

    def tick(self, pipes) -> None:
        # Spawn timer logic
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            if random.random() < 0.5:  # 50% chance to spawn
                position = self.find_safe_position(pipes)
                if position:
                    self.add_coin(*position)

        # Remove coins that are off-screen
        self.coins = [coin for coin in self.coins if coin.x > -coin.w]

        # Update remaining coins
        for coin in self.coins:
            coin.tick()

    def stop(self) -> None:
        """Stop all coins from moving"""
        for coin in self.coins:
            coin.vel_x = 0
