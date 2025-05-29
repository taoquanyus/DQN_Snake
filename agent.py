import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import mean

from dqn import DQN
from error import IllegalMoveError
from qnet import Qnet
from replay_buffer import ReplayBuffer
from snake_game import SnakeGame
from static_parameters import EPSILON, GAMMA, MAX_MEMORY, BATCH_SIZE, SHOW, SAVE
from tqdm import tqdm


class Agent:

    def __init__(self):

        self.epsilon = EPSILON  # randomness
        self.gamma = GAMMA  # discount rate
        self.memory = ReplayBuffer(MAX_MEMORY)
        self.load_model()
        self.trainer = DQN(self.model).cuda()
        # self.trainer.cuda()
        self.game = SnakeGame()
        self.batch_size = BATCH_SIZE
        self.train_score = self.model.train_score

        # self.graph_init()

    def load_model(self):
        try:
            with open('./data/epoch.txt', 'r', encoding='utf8') as fp:
                self.epoch = int(fp.readline())
                fp.close()
                self.model = torch.load(f'./data/{self.epoch}.pth')
                self.model.eval()
        except:
            self.epoch = 1
            self.model = Qnet()

    def save_model(self):
        with open('./data/epoch.txt', 'w', encoding='utf8') as fp:
            fp.write(str(self.epoch))
            fp.close()
            torch.save(self.model, f'./data/{self.epoch}.pth')

    def graph_init(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-0.5, 15.5)
        self.ax.set_ylim(-0.5, 15.5)
        self.ax.plot(0, 0)
        self.fig.canvas.draw()

    def train_long_memory(self):
        if self.memory.size() != 0:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # self.memory.append((state, action, reward, next_state, done))
        self.memory.add(state, action, reward, next_state, done)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        if self.epoch <= 1500:
            # 迭代论数小于1500时，有eplison=5%
            if random.randint(0, 19) == 0:
                return random.randint(0, 3)
        elif self.epoch <= 5000:
            # 迭代论数小于5000时，有eplison=2%
            if random.randint(0, 49) == 0:
                return random.randint(0, 3)
        elif self.epoch <= 15000:
            # 迭代论数小于5000时，有eplison=1/300
            if random.randint(0, 299) == 0:
                return random.randint(0, 3)
        state = torch.tensor(state).cuda()
        state = torch.unsqueeze(state, 0)
        prediction = self.model(state)
        move = torch.argmax(prediction).item()
        # 索引即是答案
        # 选出最大的那个索引，并用item()把它提取出来
        # 返回索引号
        return move

    def update_ui(self):
        snake_head_x = self.game.snake[-1][0]
        snake_head_y = self.game.snake[-1][1]
        point_list_x = [x for (x, y) in self.game.snake]
        point_list_y = [y for (x, y) in self.game.snake]
        plt.cla()
        self.ax.plot(point_list_x, point_list_y, 'r-', snake_head_x, snake_head_y, 'go',
                     self.game.i, self.game.j, 'bo')
        plt.draw()
        self.ax.set_xlim(-0.5, 15.5)
        self.ax.set_ylim(-0.5, 15.5)
        plt.pause(0.03)

    def train(self):
        while True:
            with tqdm(total=SAVE, desc='round %d' % (self.epoch / SAVE), colour='BLUE') as pbar:
                scores = []
                for i in range(SAVE):
                    self.game.init()
                    if self.epoch % SHOW == 0:
                        self.graph_init()
                    done = False
                    # 判断游戏是否结束
                    state = self.game.get_state()
                    # 先获得state
                    next_state = self.game.get_state()
                    while not done:
                        # Default reward for each step
                        reward = -0.1
                        state = self.game.get_state()
                        action = self.get_action(state)

                        head_old_x, head_old_y = self.game.snake[-1]
                        food_x, food_y = self.game.i, self.game.j
                        dist_old = abs(head_old_x - food_x) + abs(head_old_y - food_y)

                        try:
                            eaten = self.game.move(action)
                        except IllegalMoveError:
                            reward = -30
                            done = True
                        except TimeoutError:
                            reward = -30
                            done = True
                        else:
                            if self.epoch % SHOW == 0:
                                self.update_ui()
                            
                            if eaten:
                                reward = 50
                            else:
                                head_new_x, head_new_y = self.game.snake[-1]
                                dist_new = abs(head_new_x - food_x) + abs(head_new_y - food_y)
                                if dist_new < dist_old:
                                    reward = 2
                                elif dist_new > dist_old:
                                    reward = -3
                                # If dist_new == dist_old, reward remains -0.1

                            next_state = self.game.get_state()
                        self.train_short_memory(state, action, reward, next_state, done)
                    if self.epoch % SHOW == 0:
                        plt.pause(1)
                        plt.close()
                    scores.append(self.game.score)
                    # 每轮结束了再train long memory
                    if self.memory.size() > self.batch_size:
                        self.train_long_memory()
                    pbar.set_postfix({'episode': '%d' % self.epoch,
                                      'mean score': '%f' % np.mean(scores)})
                    pbar.update(1)
                    # print(self.epoch, self.game.score, self.game.steps)

                    if self.epoch % SAVE == 0:
                        self.save_model()

                    self.epoch += 1
                self.train_score.append(np.mean(scores))
                plt.plot(range(len(self.train_score)), self.train_score)
                plt.show()
                plt.xlabel('epoch')
                plt.ylabel('mean score')
