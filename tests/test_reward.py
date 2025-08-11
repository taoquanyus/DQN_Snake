import numpy as np
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
    reward = 40 * int(eaten)
    if not eaten:
        if new_distance < old_distance:
            reward += 1
        else:
            reward -= 1
    return reward


def test_reward_increases_when_closer_to_food():
    game = setup_game((5, 5), (5, 7))
    reward = compute_reward(game, 3)  # move right toward food
    assert reward == 1


def test_reward_decreases_when_farther_from_food():
    game = setup_game((5, 5), (5, 7))
    reward = compute_reward(game, 2)  # move left away from food
    assert reward == -1
