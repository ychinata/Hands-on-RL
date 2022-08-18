import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
# func: chapter 5.3 Sarsa Algorithm
# author: xy
# date: 2022.8.17

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
                a[i] = 1  # 记录下该动作
        return a

    def update(self, s0, a0, r, s1, a1):
        # 时序差分误差 = r_t + gamma*V(s_{t+1})-V(s_t)
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        # 更新动作价值函数
        self.Q_table[s0, a0] += self.alpha * td_error


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    """ 显示智能体的动作 """
    for i in range(env.nrow):  # 外层从上到下
        for j in range(env.ncol):  # 内层从左到右
            # self.y * self.ncol + self.x
            if (i * env.ncol + j) in disaster:  # 进入悬崖
                print('****', end=' ')
            elif (i * env.ncol + j) in end:  # 进入终点
                print('EEEE', end=' ')
            else:
                # 查找位置(i,j)的最优动作，返回的a格式形如[0,1,0,0]
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                # 打印策略，比如"ooo>"表示最优动作是向右移动，如果四个动作均为最优，则为"^v<>"
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == '__main__':
    ncol = 4  # 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1  # 学习率
    gamma = 0.9  # 折扣因子
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量
    print("\n")
    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数 50
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:  # 漫步未结束则继续执行
                    next_state, reward, done = env.step(action)
                    # next_action只用于计算，并不实际执行
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    ''' 为什么打印出来是每50条episode不是每10条? '''
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:]),
                        'ii': i_episode,
                        'i': i
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))  # len = 500
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()
    # 打印最终策略
    action_meaning = ['^', 'v', '<', '>']  # 分别表示[向上,向下,向左,向右]移动
    print('Sarsa算法最终收敛得到的策略为：')
    # 37-46是悬崖
    cliff_bgn = (nrow-1)*ncol+1
    goal_end = nrow*ncol-1
    print_agent(agent, env, action_meaning, list(range(cliff_bgn, goal_end)), [goal_end])
