import random
from collections import deque

import numpy as np

from error import IllegalMoveError


class SnakeGame():
    def __init__(self):
        self.init()

    def init(self):
        # 初始化棋盘
        self.score = 0
        self.steps = 0
        self.time = 100
        self.snake_board = np.zeros((16, 16), dtype=np.float32)
        self.snake = deque()
        self.p = random.randint(0, 15)
        self.q = random.randint(0, 15)
        self.snake_board[self.p][self.q] = 1
        # 蛇身的位置返回1，空白位置返回0
        self.snake.append((self.p, self.q))
        self.DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
        self.place_food()
        self.eaten = False

    def place_food(self):
        # Ensure that food does not spawn on the snake's body
        while True:
            self.i = random.randint(0, 15)
            self.j = random.randint(0, 15)
            if (self.i, self.j) not in self.snake:
                break

    def get_state(self):
        incoord = lambda m, n: m in range(0, 16) and n in range(0, 16)
        # 判断两个坐标是否都在incoord中
        grid = lambda m, n: self.snake_board[m + self.p][n + self.q] if incoord(m + self.p, n + self.q) else 0
        # 如果不在棋盘中，返回0，如果在棋盘中，返回snake_board[m + self.p][n + self.q]
        return np.array([
            grid(-3,0),
            grid(-2, -1), grid(-2, 0), grid(-2, 1),
            grid(-1,-2), grid(-1, -1), grid(-1, 0), grid(-1, 1), grid(-1, 2),
            grid(0,-3), grid(0, -2), grid(0, -1), grid(0, 0), grid(0, 1), grid(0, 2), grid(0,3),
            grid(1,-2), grid(1, -1), grid(1, 0), grid(1, 1), grid(1,2),
            grid(2,-1), grid(2, 0), grid(2,1),
            grid(3,0),
            self.p, self.q, 15 - self.p, 15 - self.q,
                            self.p - self.i, self.q - self.j, self.time, self.score
        ], dtype=np.float32)

    def move(self, direction):
        # move
        last_p, last_q = self.snake[-2] if len(self.snake) > 1 else self.snake[-1]
        self.p, self.q = self.snake[-1]
        if len(self.snake) >= 2:
            last_direction = [self.p - last_p, self.q - last_q]
            if last_direction[0] * self.DIRECTIONS[direction][0] + last_direction[1] * self.DIRECTIONS[direction][
                1] == -1:
                # 表示如果倒着走直接直走
                self.p += last_direction[0]
                self.q += last_direction[1]
            else:
                self.p += self.DIRECTIONS[direction][0]
                self.q += self.DIRECTIONS[direction][1]
        else:
            self.p += self.DIRECTIONS[direction][0]
            self.q += self.DIRECTIONS[direction][1]
        self.time -= 1
        if not (0 <= self.p < 16 and 0 <= self.q < 16):
            # print("die from wall collide")
            raise IllegalMoveError

        if self.time < 0:
            # print("die from time is up")
            raise TimeoutError
        self.snake.append((self.p, self.q))
        if not self.eaten: self.snake.popleft()

        # judge food
        self.eaten = False
        food_pos = (self.i, self.j)
        for a in self.snake:
            if a == food_pos:
                self.eaten = True
                self.score += 1
                self.time = 100 + self.score
                self.place_food()
                break

        # judge if self collide
        i = 2 if self.eaten else 1
        self.snake_board = np.zeros((16, 16), dtype=np.float32)
        for (p, q) in self.snake:
            if self.snake_board[p][q] != 0:
                # print("die from self collide")
                raise IllegalMoveError
            self.snake_board[p][q] = i
            i += 1

        self.steps += 1
        return self.eaten