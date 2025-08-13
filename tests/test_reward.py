import numpy as np
import pytest
from collections import deque

from snake_game import SnakeGame


def setup_game(head, food):
    game = SnakeGame()
    game.p, game.q = head
    game.snake = deque([head])
    game.snake_board = np.zeros((16, 16), dtype=np.float32)
    game.snake_board[head[0]][head[1]] = 1
    game.i, game.j = food
    return game


def compute_reward(game, action):
    old_distance = abs(game.p - game.i) + abs(game.q - game.j)
    eaten = game.move(action)
    new_distance = abs(game.p - game.i) + abs(game.q - game.j)
    distance_change = old_distance - new_distance
    reward = 40 * int(eaten)
    if not eaten:
        reward += distance_change - 0.1
    return reward


def test_reward_increases_when_closer_to_food():
    game = setup_game((5, 5), (5, 7))
    reward = compute_reward(game, 3)  # move right toward food
    assert reward == pytest.approx(0.9)


def test_reward_decreases_when_farther_from_food():
    game = setup_game((5, 5), (5, 7))
    reward = compute_reward(game, 2)  # move left away from food
    assert reward == pytest.approx(-1.1)
