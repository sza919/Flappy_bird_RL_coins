import pygame

from ..utils import GameConfig
from .entity import Entity


class Score(Entity):
    def __init__(self, config: GameConfig) -> None:
        super().__init__(config)
        self.y = self.config.window.height * 0.1
        self.score = 0
        self.coins_collected = 0

    def reset(self) -> None:
        self.score = 0
        self.coins_collected = 0

    def add_coins(self, count: int = 1) -> None:
        self.coins_collected += count
        self.score += count * 5

    def add(self) -> None:
        self.score += 1
        self.config.sounds.point.play()

    @property
    def rect(self) -> pygame.Rect:
        score_digits = [int(x) for x in list(str(self.score))]
        images = [self.config.images.numbers[digit] for digit in score_digits]
        w = sum(image.get_width() for image in images)
        x = (self.config.window.width - w) / 2
        h = max(image.get_height() for image in images)
        return pygame.Rect(x, self.y, w, h)

    def draw(self) -> None:
        """displays score in center of screen"""
        score_digits = [int(x) for x in list(str(self.score))]
        images = [self.config.images.numbers[digit] for digit in score_digits]
        digits_width = sum(image.get_width() for image in images)
        x_offset = (self.config.window.width - digits_width) / 2

        for image in images:
            self.config.screen.blit(image, (x_offset, self.y))
            x_offset += image.get_width()

        if hasattr(self.config.images, "coin") and self.coins_collected > 0:
            coin_icon = self.config.images.coin
            coin_y = self.y + 30  # Show below score
            self.config.screen.blit(coin_icon, (10, coin_y))

            coin_digits = [int(x) for x in list(str(self.coins_collected))]
            coin_images = [
                self.config.images.numbers[digit] for digit in coin_digits
            ]
            x_offset = 20 + coin_icon.get_width()

            for image in coin_images:
                self.config.screen.blit(image, (x_offset, coin_y))
                x_offset += image.get_width()
