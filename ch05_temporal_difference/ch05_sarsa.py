import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
# func: chapter 5.3 Sarsa Algorithm
# author: xy
# date: 2022.8.17
# 把环境修改为3*3看看？


# 悬崖漫步环境
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow  # 边界
        self.ncol = ncol  # 边界
        # 起点在(x=0, y=nrow - 1)
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    """ 执行漫步的动作 """
    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。
        # 坐标系原点(0,0)定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        # 将位置限置在环境内，x:[0, ncol-1], y:[0, nrow-1]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1  # 智能体每走一步的奖励是-1
        done = False
        # 下一个位置在悬崖或者目标,标志游戏结束
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:  # 目标是(x=ncol-1, y=nrow-1)
                reward = -100  # 掉入悬崖的奖励是-100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        # epsilon的概率从动作空间中随机采取一个动作
        if np.random.random() < self.epsilon:  # np.random.random()取值范围：[0,1]
            action = np.random.randint(self.n_action)  # 在[0,n_action-1]范围随机取一个整数
        # (1-epsilon)的概率采用动作价值最大的动作
        else:
            # 为什么用argmax而不是max?
            # argmax输出的是自变量action, max输出的是因变量q
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]  # a = [0,...,0]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

