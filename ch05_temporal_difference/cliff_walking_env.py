# func: chapter 5.3 Sarsa Algorithm
# author: xy
# date: 2022.8.20

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
