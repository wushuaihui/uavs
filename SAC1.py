import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from uav_2d import EnvWrapper
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math as m
import os
import math
from torch.distributions import Normal  # 添加Normal类的导入

# 设置OpenMP线程数以避免Intel libiomp和LLVM libomp库冲突
# 当同时加载这两个不兼容的OpenMP库时，可能导致程序随机崩溃或死锁
# 将OMP_NUM_THREADS设置为1可以缓解此问题
# 同时设置其他相关环境变量以进一步减少冲突可能性
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'
# 添加更多环境变量设置以进一步减少OpenMP库冲突
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# 设置MKL线程层为GNU以避免Intel和LLVM OpenMP库冲突
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# 导入绘图模块
import draw as d

# 导入拒止区域处理器
from denied_area_handler import DeniedAreaHandler

# SAC算法相关超参数
GAMMA = 0.99         # 折扣因子
TAU = 0.005          # 软更新参数
LR_ACTOR = 3e-4      # Actor网络学习率
LR_CRITIC = 3e-4     # Critic网络学习率
LR_ALPHA = 3e-4      # Alpha学习率
TARGET_ENTROPY_RATIO = 0.98  # 目标熵比率
BATCH_SIZE = 512     # 增加批处理大小以更好地利用GPU
START_UPD_SAMPLES = 1000  # 开始更新前需要的样本数
PI = 3.14159265358979323846  # 圆周率常量


class Point2D:
    """二维点类，用于表示坐标位置
    该类用于存储二维空间中的点坐标，如无人机位置、目标位置、障碍物位置等
    """

    def __init__(self):
        self.x = complex()  # x坐标，表示水平方向位置
        self.y = complex()  # y坐标，表示垂直方向位置


# 定义失联无人机和邻近无人机
# 用于追踪失联无人机的位置预测和历史位置记录
lost_uav = Point2D()  # 失联无人机当前位置
past_uav = Point2D()  # 失联无人机上一时刻位置
lost_uav.x = 10.0  # 初始化失联无人机x坐标
lost_uav.y = 45  # 初始化失联无人机y坐标，失联无人机的初始位置


def l2norm(x, y):
    """
    计算二维欧几里得距离
    :param x: x方向距离
    :param y: y方向距离
    :return: 欧几里得距离
    """
    return m.sqrt(x * x + y * y)


def real_done(done):
    """
    检查所有智能体是否都完成任务
    :param done: 各智能体完成状态字典
    :return: 所有智能体都完成返回True，否则返回False
    """
    for v in done.values():
        if not v:
            return False
    return True


def includedangle(x1, y1, x2, y2):
    """
    计算从点(x2,y2)到点(x1,y1)的角度（以x轴正方向为0度）
    使用numpy的arctan2函数直接计算，提高效率

    :param x1: 目标点x坐标
    :param y1: 目标点y坐标
    :param x2: 无人机x坐标
    :param y2: 无人机y坐标
    :return: 角度值（弧度），范围在[-π, π]之间
    """
    dx = x1 - x2
    dy = y1 - y2
    # 直接使用numpy的arctan2函数计算角度，自动处理所有象限
    angle = np.arctan2(dy, dx)
    return angle


# 设置计算设备（GPU或CPU）
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

print(T.cuda.is_available())  # 打印CUDA是否可用

# 解析命令行参数
parser = argparse.ArgumentParser(description='Input n_agents and main folder')
parser.add_argument('--agents', type=int, default=5)  # 无人机数量，默认为5
parser.add_argument('--folder', type=str)  # 主文件夹路径
parser.add_argument('--global_', type=str, default="GLOBAL")  # 全局视角设置
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/SAC1/')  # 检查点保存目录
parser.add_argument('--checkpoint', type=str, default=None, help='要加载的检查点文件')  # 添加checkpoint参数

args = parser.parse_args()

N_AGENTS = args.agents  # 无人机数量
MAIN_FOLDER = args.folder  # 主文件夹路径

# 动作分布对数标准差的范围限制
# 用于限制策略网络输出的动作分布的标准差范围，避免数值不稳定
LOG_STD_MIN = -20  # 对数标准差下限，防止标准差过小导致的数值问题
LOG_STD_MAX = 2  # 对数标准差上限，防止标准差过大导致的探索过度


class ActorNetwork(nn.Module):
    """
    Actor网络类，用于输出策略分布
    根据当前状态输出动作的概率分布
    在SAC算法中，Actor网络负责学习一个随机策略，将状态映射到动作分布
    适配48维观测空间
    """

    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        """
        初始化Actor网络
        :param alpha: 学习率，控制参数更新的步长
        :param state_dim: 状态空间维度，输入状态的特征数量（48维）
        :param action_dim: 动作空间维度，输出动作的特征数量
        :param fc1_dim: 第一个全连接层的输出维度
        :param fc2_dim: 第二个全连接层的输出维度
        """
        super(ActorNetwork, self).__init__()

        # 全连接层处理状态数据（移除了CNN部分，因为我们不再使用图像数据）
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        # 第一层层归一化：对fc1的输出进行归一化，加速训练
        self.ln1 = nn.LayerNorm(fc1_dim)
        # 第二层全连接层：将fc1_dim维特征映射到fc2_dim维特征空间
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        # 第二层层归一化：对fc2的输出进行归一化
        self.ln2 = nn.LayerNorm(fc2_dim)

        # 动作均值输出层
        self.mean_linear = nn.Linear(fc2_dim, action_dim)
        # 动作对数标准差输出层
        self.log_std_linear = nn.Linear(fc2_dim, action_dim)

        # 优化器设置：使用Adam优化器，学习率为alpha
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)  # 将网络移到指定设备（GPU或CPU）

    def forward(self, state):
        """
        前向传播函数，根据状态计算动作分布参数 - 高性能版
        """
        # 全连接层处理
        # 第一层：线性变换 -> 层归一化 -> ReLU激活
        x = F.relu(self.ln1(self.fc1(state)))
        # 第二层：线性变换 -> 层归一化 -> ReLU激活
        x = F.relu(self.ln2(self.fc2(x)))

        # 计算均值和对数标准差
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = T.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        """
        从策略分布中采样动作 - 优化版
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 优化采样和概率计算
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = T.tanh(x_t)

        # 向量化计算对数概率，避免中间变量
        # 增加数值稳定性检查
        log_prob_normal = normal.log_prob(x_t)
        # 确保action值在有效范围内以避免数值问题
        action_clamped = T.clamp(action, -0.999, 0.999)
        log_prob_tanh = T.log(1 - action_clamped.pow(2) + 1e-6)
        log_prob = log_prob_normal - log_prob_tanh
        # 检查log_prob的维度，确保在正确的维度上求和
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum().unsqueeze(0).unsqueeze(1)

        return action, log_prob, mean

    def save_checkpoint(self, checkpoint_file):
        """保存网络检查点"""
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        """加载网络检查点"""
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    """
    Critic网络类，用于评估状态-动作对的Q值
    在SAC算法中，Critic网络用于评估给定状态下采取某个动作的价值（Q值）
    SAC使用两个Critic网络来减少过估计问题
    适配48维观测空间
    """

    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()

        # 将状态和动作拼接后输入网络
        # 输入维度：状态维度 + 动作维度
        total_input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(total_input_dim, fc1_dim)
        # 第一层层归一化：对fc1的输出进行归一化，加速训练
        self.ln1 = nn.LayerNorm(fc1_dim)
        # 第二层全连接层：将fc1_dim维特征映射到fc2_dim维特征空间
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        # 第二层层归一化：对fc2的输出进行归一化
        self.ln2 = nn.LayerNorm(fc2_dim)
        # Q值输出层：将fc2_dim维特征映射到标量Q值
        self.q = nn.Linear(fc2_dim, 1)  # 输出Q值

        # 优化器设置：使用Adam优化器，学习率为beta
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # 将网络移到指定设备（GPU或CPU）
        self.to(device)

    def forward(self, state, action):
        """
        前向传播，计算状态-动作对的Q值 - 高性能版
        """
        # 合并状态和动作
        combined = T.cat([state, action], dim=1)

        # 全连接层处理
        # 第一层：线性变换 -> 层归一化 -> ReLU激活
        x = F.relu(self.ln1(self.fc1(combined)))
        # 第二层：线性变换 -> 层归一化 -> ReLU激活
        x = F.relu(self.ln2(self.fc2(x)))
        # 输出层：线性变换得到Q值
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        """保存网络检查点"""
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        """加载网络检查点"""
        self.load_state_dict(T.load(checkpoint_file))


class ReplayBuffer:
    """
    经验回放缓冲区类
    用于存储智能体与环境交互的经验，支持随机采样以打破数据相关性
    经验回放是深度强化学习中的关键技术，可以提高样本效率并减少过拟合
    通过存储和重放过去的经验，使得智能体可以多次利用相同的经验进行学习
    适配包含雷达数据和拒止区域状态标记的高维观测空间
    """

    def __init__(self, max_size, state_dim, action_dim, batch_size):
        """
        初始化经验回放缓冲区
        :param max_size: 最大存储大小，缓冲区能存储的最大转移数量
        :param state_dim: 状态空间维度（48维）
        :param action_dim: 动作空间维度
        :param batch_size: 批处理大小，每次采样时返回的样本数量
        """
        # 确保状态维度为48维
        self.effective_state_dim = state_dim  # 状态维度固定为48维

        self.mem_size = max_size  # 最大存储大小
        self.batch_size = batch_size  # 批处理大小
        self.mem_cnt = 0  # 当前存储数量

        # 初始化存储数组，使用float32减少内存占用
        # 状态记忆数组：存储智能体观察到的状态信息，形状为(最大存储大小, 状态维度)
        self.state_memory = np.zeros((max_size, self.effective_state_dim), dtype=np.float32)
        # 动作记忆数组：存储智能体采取的动作，形状为(最大存储大小, 动作维度)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        # 奖励记忆数组：存储执行动作后获得的奖励，形状为(最大存储大小,)
        self.reward_memory = np.zeros((max_size,), dtype=np.float32)
        # 下一状态记忆数组：存储执行动作后环境转移到的新状态，形状为(最大存储大小, 状态维度)
        self.next_state_memory = np.zeros((max_size, self.effective_state_dim), dtype=np.float32)
        # 终止标志记忆数组：存储是否到达终止状态的标志，形状为(最大存储大小,)
        self.terminal_memory = np.zeros((max_size,), dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        """
        存储转移样本到经验回放缓冲区 - 优化版
        优化状态处理逻辑，减少内存操作
        """
        # 计算当前存储位置，使用模运算实现循环缓冲区
        mem_idx = self.mem_cnt % self.mem_size

        # 优化版状态处理 - 使用更高效的向量化操作
        # 处理当前状态（精确处理48维状态）
        if len(state) != self.effective_state_dim:
            if len(state) < self.effective_state_dim:
                # 直接在目标数组上操作，避免中间数组创建
                self.state_memory[mem_idx, :len(state)] = state
                self.state_memory[mem_idx, len(state):] = 0  # 剩余部分设为0
            else:
                self.state_memory[mem_idx] = state[:self.effective_state_dim]
        else:
            self.state_memory[mem_idx] = state

        # 处理下一状态（精确处理48维状态）
        if len(state_) != self.effective_state_dim:
            if len(state_) < self.effective_state_dim:
                self.next_state_memory[mem_idx, :len(state_)] = state_
                self.next_state_memory[mem_idx, len(state_):] = 0
            else:
                self.next_state_memory[mem_idx] = state_[:self.effective_state_dim]
        else:
            self.next_state_memory[mem_idx] = state_

        # 存储动作、奖励和终止标志
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.terminal_memory[mem_idx] = done

        # 更新存储计数器
        self.mem_cnt += 1

    def sample_buffer(self):
        """
        从缓冲区中随机采样一个批次 - 优化版
        减少不必要的类型转换和内存复制
        """
        # 确定当前可用的样本数量
        mem_len = min(self.mem_cnt, self.mem_size)

        # 优化随机采样，避免创建额外的索引数组
        batch_indices = np.random.randint(0, mem_len, self.batch_size)

        # 直接返回数组视图而非复制，numpy操作已经优化
        # 注意：由于我们在初始化时已经使用了float32，这里不需要再次转换
        states = self.state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        states_ = self.next_state_memory[batch_indices]
        terminals = self.terminal_memory[batch_indices]

        return states, actions, rewards, states_, terminals

    def n_samples(self):
        """返回当前存储的样本数"""
        return min(self.mem_cnt, self.mem_size)

    def ready(self):
        """检查缓冲区是否已准备好进行采样"""
        return self.mem_cnt >= self.batch_size

    def get_state(self):
        """获取缓冲区的当前状态，用于保存检查点 - 简化版"""
        # 只保存已使用的部分，减少内存占用
        mem_len = min(self.mem_cnt, self.mem_size)
        return {
            'mem_cnt': self.mem_cnt,
            'state_memory': self.state_memory[:mem_len],
            'action_memory': self.action_memory[:mem_len],
            'reward_memory': self.reward_memory[:mem_len],
            'next_state_memory': self.next_state_memory[:mem_len],
            'terminal_memory': self.terminal_memory[:mem_len]
        }

    def load_state(self, state_dict):
        """从状态字典加载缓冲区状态，用于恢复检查点 - 优化版"""
        self.mem_cnt = state_dict['mem_cnt']
        mem_len = min(self.mem_cnt, self.mem_size)

        # 高效加载，避免不必要的复制和类型转换
        if mem_len > 0:
            self.state_memory[:mem_len] = state_dict['state_memory']
            self.action_memory[:mem_len] = state_dict['action_memory']
            self.reward_memory[:mem_len] = state_dict['reward_memory']
            self.next_state_memory[:mem_len] = state_dict['next_state_memory']
            self.terminal_memory[:mem_len] = state_dict['terminal_memory']


class SAC:
    """
    SAC算法实现类（Soft Actor-Critic）
    SAC是一种基于最大熵强化学习的算法，通过最大化期望奖励和策略熵的加权和来学习策略
    SAC的主要特点：
    1. 策略是随机的（与TD3的确定性策略不同）
    2. 使用最大熵框架，鼓励探索
    3. 使用双Q网络减少过估计问题
    4. 使用目标网络稳定训练
    """

    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir, gamma=0.99, tau=0.005,
                 max_size=1000000, batch_size=256):
        """
        初始化SAC算法，适配包含雷达数据和拒止区域状态标记的观测空间
        
        :param alpha: Actor网络学习率
        :param beta: Critic网络学习率
        :param state_dim: 状态空间维度（包含雷达数据和拒止区域状态标记）
        :param action_dim: 动作空间维度
        :param actor_fc1_dim: Actor网络第一层全连接层维度
        :param actor_fc2_dim: Actor网络第二层全连接层维度
        :param critic_fc1_dim: Critic网络第一层全连接层维度
        :param critic_fc2_dim: Critic网络第二层全连接层维度
        :param ckpt_dir: 检查点保存目录
        :param gamma: 折扣因子
        :param tau: 软更新参数
        :param max_size: 经验回放缓冲区最大大小
        :param batch_size: 批处理大小
        """
        # 算法超参数
        self.gamma = gamma  # 折扣因子，用于计算未来奖励的现值
        self.tau = tau  # 软更新参数，控制目标网络更新速度
        self.batch_size = batch_size  # 批处理大小
        self.checkpoint_dir = ckpt_dir  # 检查点目录，用于保存和加载模型

        # 确保状态维度为48维
        effective_state_dim = state_dim  # 状态维度固定为48维

        # 初始化网络
        # Actor网络：根据状态输出动作分布
        self.actor = ActorNetwork(alpha=alpha, state_dim=effective_state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)

        # 双重Critic网络：评估状态-动作对的Q值，减少过估计
        self.critic1 = CriticNetwork(beta=beta, state_dim=effective_state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic2 = CriticNetwork(beta=beta, state_dim=effective_state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        # 初始化目标网络
        # 目标Critic网络：用于计算目标Q值
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=effective_state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=effective_state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(max_size=max_size, state_dim=effective_state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        # 自动熵调节参数
        self.target_entropy = -action_dim  # 修复target_entropy计算
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        # 初始化目标网络参数，使用tau=1.0进行硬更新
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        """
        软更新目标网络参数 - 优化版
        """
        tau = self.tau if tau is None else tau

        # 批量更新目标网络，使用inplace操作减少内存
        with T.no_grad():
            # 合并更新两个目标网络
            for (critic1_p, target1_p), (critic2_p, target2_p) in zip(
                    zip(self.critic1.parameters(), self.target_critic1.parameters()),
                    zip(self.critic2.parameters(), self.target_critic2.parameters())
            ):
                target1_p.data.mul_(1. - tau).add_(tau * critic1_p.data)
                target2_p.data.mul_(1. - tau).add_(tau * critic2_p.data)

    def remember(self, state, action, reward, state_, done):
        """
        将转移样本存储到经验回放缓冲区
        这是与环境交互后存储经验的接口方法
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param state_: 下一状态
        :param done: 是否结束标志
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observations, uavs, obs, episode, target_pos, w, v, n_agents, is_training, 
                      uav_denied_status=None, uav_pointclouds=None, denied_area_handler=None):
        """
        选择动作 - 改进版，支持拒止区域处理和启发式算法
        """
        actions = []
        
        # 获取有效的无人机ID列表（存活的无人机）
        valid_uav_ids = list(uavs.keys()) if isinstance(uavs, dict) else list(range(len(uavs)))
        # 处理每个无人机的动作
        for i in range(n_agents):
            # 检查无人机是否在拒止区域内
            in_denied_area = uav_denied_status.get(i, False) if uav_denied_status else False
            
            # 检查无人机是否存活（存在于uavs中）
            is_alive = i in valid_uav_ids
            
            if in_denied_area and uav_pointclouds and denied_area_handler and is_alive:
                # 在拒止区域内，使用聚类算法进行跟随
                pointcloud = uav_pointclouds.get(i, None)

                # 获取无人机在uavs中的索引
                uav_index = valid_uav_ids.index(i) if isinstance(uavs, dict) else i
                if pointcloud is not None:
                    uav_pos = (uavs[i][0], uavs[i][1]) if isinstance(uavs, dict) else (uavs[uav_index][0], uavs[uav_index][1])
                    # 使用GWO算法进行跟随决策
                    from gwo import gwo_follow
                    
                    # 从点云数据中提取友机位置
                    target_positions = self.extract_uav_positions_from_pointcloud(pointcloud)

                    
                    # 获取障碍物位置（从环境中获取）
                    # 在拒止区域内不考虑障碍物，因为无人机无法获取环境地图
                    obstacle_positions = []
                    
                    # 如果有可跟随的目标，使用GWO算法计算最优跟随位置
                    if target_positions and uav_pos:
                        # target_positions是相对于当前无人机雷达的局部坐标
                        # uav_pos是当前无人机位置（基于惯性导航或最后已知位置的估算值）
                        best_follow_pos = gwo_follow(target_positions, uav_pos)
                        # 计算到最优跟随位置的动作
                        # best_follow_pos是局部坐标，相对于雷达坐标系
                        # 传递[0, 0]作为当前位置，best_follow_pos作为目标位置给calculate_follow_action
                        # 因为calculate_follow_action需要在同一局部坐标系中的相对位置
                        action = self.calculate_follow_action([0, 0], best_follow_pos, v[uav_index], w[uav_index])
                    else:
                        # 如果没有目标，使用原有的处理方法
                        action = denied_area_handler.process_denied_area_uav(
                            i, pointcloud, uav_pos, v[uav_index], w[uav_index], is_training
                        )
                    actions.append(action)
                else:
                    # 如果没有点云数据，保持当前状态
                    actions.append([0.0, 0.0])
            elif is_alive:  # 只为存活的无人机计算动作
                # 在拒止区域外，使用DRL或启发式算法
                # 生成0-1之间的随机数，用于决定使用启发式策略还是神经网络策略
                r = random.uniform(0, 1)
                
                # 获取无人机在uavs中的索引
                uav_index = valid_uav_ids.index(i) if isinstance(uavs, dict) else i
                
                # 一定概率使用启发式策略
                # 条件：随机数小于0.3且episode小于2000且无人机数量等于观测到的无人机数量
                if r < 0.3 and episode < 2000 and n_agents == len(uavs):
                    # 启发式策略计算
                    avgv = 0  # 平均速度初始化
                    wt = [0.0] * len(uavs)  # 无人机朝向目标的角度差
                    wo = [0.0] * len(uavs)  # 无人机朝向障碍物的角度
                    lo = [0.0] * len(uavs)  # 无人机到障碍物的距离
                    action = [0.0] * len(uavs)  # 最终动作
                    turn = False  # 是否需要避障标志
                    a = [0.0] * len(uavs)  # 朝向目标的角度调整项
                    b = [0.0] * len(uavs)  # 避障角度调整项
                    c = [0.0] * len(uavs)  # 保持队形的角度调整项

                    # 计算每个无人机朝向目标的角度差
                    for j in range(len(uavs)):
                        # 计算无人机朝向目标点的角度
                        uav_pos_j = (uavs[j][0], uavs[j][1]) if isinstance(uavs, dict) else (uavs[j][0], uavs[j][1])
                        wt[j] = includedangle(target_pos.x, target_pos.y, uav_pos_j[0], uav_pos_j[1])
                        # 根据当前航向和目标方向的差异计算角度调整量
                        if w[j] > wt[j]:
                            a[j] = (wt[j] - w[j]) / (PI)
                        else:
                            a[j] = (wt[j] - w[j]) / (PI)

                    # 检查是否有障碍物接近，判断是否需要避障
                    # 修复：确保只访问有效的障碍物索引
                    valid_obs_count = min(len(obs), 2)  # 只使用前2个障碍物
                    for j in range(len(uavs)):
                        lo[j] = [0.0] * valid_obs_count
                        for k in range(valid_obs_count):
                            # 计算无人机到障碍物的距离
                            uav_pos_j = (uavs[j][0], uavs[j][1]) if isinstance(uavs, dict) else (uavs[j][0], uavs[j][1])
                            lo[j][k] = l2norm(obs[k].x - uav_pos_j[0], obs[k].y - uav_pos_j[1])
                            # 如果距离小于50单位，则需要避障
                            if lo[j][k] < 50:
                                turn = True

                    # 如果需要避障，则计算避障策略
                    if turn:
                        # 计算避障策略
                        for j in range(len(uavs)):
                            wo[j] = [0.0] * valid_obs_count
                            for k in range(valid_obs_count):
                                # 计算无人机朝向障碍物的角度
                                uav_pos_j = (uavs[j][0], uavs[j][1]) if isinstance(uavs, dict) else (uavs[j][0], uavs[j][1])
                                wo[j][k] = includedangle(obs[k].x, obs[k].y, uav_pos_j[0], uav_pos_j[1])
                                # 根据当前航向和障碍物方向的差异计算避障角度调整量
                                if w[j] > wo[j][k]:
                                    k_val = (w[j] - wo[j][k])
                                    b[j] = (PI - k_val) / (PI)
                                else:
                                    k_val = (wo[j][k] - w[j])
                                    b[j] = -(PI - k_val) / (PI)

                    # 计算平均速度和角速度，用于保持队形
                    avguavw = 0.0
                    valid_uav_count = 0  # 实际有效的无人机数量（排除拒止区域内的无人机）
                    for j in range(len(uavs)):
                        # 排除已进入拒止区域的无人机
                        uav_index = valid_uav_ids[j] if isinstance(uavs, dict) else j
                        if not uav_denied_status.get(uav_index, False):
                            avgv = avgv + v[j]
                            avguavw = avguavw + w[j]
                            valid_uav_count += 1
                    
                    # 只有当有有效的无人机时才计算平均值
                    if valid_uav_count > 0:
                        avguavw = avguavw / valid_uav_count
                        avgv = avgv / valid_uav_count
                    else:
                        # 如果所有无人机都进入拒止区域，则使用默认值
                        avguavw = sum(w) / len(w) if w else 0.0
                        avgv = sum(v) / len(v) if v else 2.0
                    
                    d = [0.0] * len(uavs)  # 速度调整项

                    # 计算角速度调整项，使无人机保持队形
                    for j in range(len(uavs)):
                        # 排除已进入拒止区域的无人机
                        uav_index = valid_uav_ids[j] if isinstance(uavs, dict) else j
                        if not uav_denied_status.get(uav_index, False):
                            if w[j] > avguavw:
                                c[j] = (avguavw - w[j]) / (PI)
                            else:
                                c[j] = (avguavw - w[j]) / (PI)
                    
                    # 添加额外的群体聚集行为
                    # 鼓励无人机向群体中心移动以保持编队
                    group_center_x = 0.0
                    group_center_y = 0.0
                    valid_group_count = 0
                    
                    for j in range(len(uavs)):
                        # 排除已进入拒止区域的无人机
                        uav_index = valid_uav_ids[j] if isinstance(uavs, dict) else j
                        if not uav_denied_status.get(uav_index, False):
                            uav_pos_j = (uavs[j][0], uavs[j][1]) if isinstance(uavs, dict) else (uavs[j][0], uavs[j][1])
                            group_center_x += uav_pos_j[0]
                            group_center_y += uav_pos_j[1]
                            valid_group_count += 1
                    
                    # 如果有有效的群体成员，计算群体中心并鼓励向其移动
                    if valid_group_count > 1:  # 至少需要2架无人机形成群体
                        group_center_x /= valid_group_count
                        group_center_y /= valid_group_count
                        
                        # 对每个无人机计算向群体中心移动的倾向
                        for j in range(len(uavs)):
                            # 排除已进入拒止区域的无人机
                            uav_index = valid_uav_ids[j] if isinstance(uavs, dict) else j
                            if not uav_denied_status.get(uav_index, False):
                                uav_pos_j = (uavs[j][0], uavs[j][1]) if isinstance(uavs, dict) else (uavs[j][0], uavs[j][1])
                                # 计算到群体中心的距离
                                dist_to_center = l2norm(uav_pos_j[0] - group_center_x, uav_pos_j[1] - group_center_y)
                                # 如果距离群体中心较远，添加向中心移动的倾向
                                if dist_to_center > 30:  # 距离超过30单位时开始调整
                                    # 计算向群体中心的方向
                                    center_angle = includedangle(group_center_x, group_center_y, uav_pos_j[0], uav_pos_j[1])
                                    angle_diff = center_angle - w[j]
                                    # 规范化角度差
                                    while angle_diff > PI:
                                        angle_diff -= 2 * PI
                                    while angle_diff < -PI:
                                        angle_diff += 2 * PI
                                    
                                    # 添加向群体中心移动的倾向（权重较小以避免过度影响）
                                    group_tendency = 0.3 * angle_diff / PI
                                    c[j] += group_tendency

                    # 计算速度调整项，使无人机保持队形
                    for j in range(len(uavs)):
                        # 排除已进入拒止区域的无人机
                        uav_index = valid_uav_ids[j] if isinstance(uavs, dict) else j
                        if not uav_denied_status.get(uav_index, False):
                            if v[j] > avgv:
                                d[j] = (avgv - v[j])
                            else:
                                d[j] = (avgv - v[j])

                    sum_vals = [0.0] * len(uavs)

                    # 限制调整值范围，防止动作过大
                    for j in range(len(uavs)):
                        if d[j] > 1:
                            d[j] = 1
                        if d[j] < -1:
                            d[j] = -1

                    # 综合计算动作，将目标导向、避障和队形保持三个因素结合起来
                    for j in range(len(uavs)):
                        sum_vals[j] = a[j] + b[j] + c[j]
                        # 限制角度调整范围
                        if sum_vals[j] > 1:
                            sum_vals[j] = 1
                        if sum_vals[j] < -1:
                            sum_vals[j] = -1

                    # 构造最终动作数组
                    for j in range(len(uavs)):
                        action[j] = [sum_vals[j], d[j]]
                    
                    actions.append(action[uav_index])
                else:
                    # 使用神经网络选择动作
                    state = T.tensor(observations[i], dtype=T.float).to(device)
                    if is_training:
                        action, _, _ = self.actor.sample(state)
                    else:
                        mean, _ = self.actor.forward(state)
                        action = T.tanh(mean)  # 在测试模式下直接使用均值
                    action_np = action.cpu().detach().numpy()[0]
                    actions.append(action_np)
            else:
                # 对于坠毁的无人机，添加默认动作
                actions.append([0.0, 0.0])
                
        return actions

    def extract_uav_positions_from_pointcloud(self, pointcloud):
        """
        从点云数据中提取无人机位置
        参考点云数据结构：(N, 4) [x, y, z, reflectance]
        :param pointcloud: 点云数据 (N, 4) [x, y, z, reflectance]
        :return: 无人机位置列表 [(x, y), ...]
        """
        # 检查点云数据是否为空
        if pointcloud is None or len(pointcloud) == 0:
            return []
            
        # 提取高反射率点（无人机反射率通常在0.5以上）
        # 阈值0.5： 区分无人机（高反射率）和环境障碍物（低反射率）
        # 无人机反射率范围通常在0.8-1.0之间，障碍物反射率范围通常在0.1-0.3之间
        drone_mask = pointcloud[:, 3] >= 0.5  # 使用标准阈值0.5来识别无人机
        drone_points = pointcloud[drone_mask]
        
        # 如果无人机点不足3个，则无法进行有效聚类
        if len(drone_points) < 3:  
            return []
            
        try:
            # 使用K-means聚类识别无人机位置
            from sklearn.cluster import KMeans
            
            # 根据实际检测到的无人机点数量智能确定聚类数
            max_possible_clusters = len(drone_points) // 30
            n_clusters = min(8, max_possible_clusters, len(drone_points))  # 最多识别8架无人机
            
            # 确保至少有1个聚类
            n_clusters = max(1, n_clusters)
            
            # 使用K-means算法对无人机点进行聚类
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            # 只使用x,y坐标进行聚类，忽略z坐标和反射率
            kmeans.fit(drone_points[:, :2])  
            
            # 获取聚类中心作为无人机位置
            centers = kmeans.cluster_centers_
            
            # 转换为实际坐标列表 [(x, y), ...]
            uav_positions = [(center[0], center[1]) for center in centers]

                
            return uav_positions
        except Exception as e:
            print(f"聚类过程中出现错误: {e}")
            return []

    def calculate_follow_action(self, own_position, target_position, own_velocity, own_heading):
        """
        计算跟随动作
        根据目标位置计算角度调整和速度调整
        
        参数:
        own_position: 自身位置 (x, y) - 世界坐标或局部坐标
        target_position: 目标位置 (x, y) - 与own_position在同一坐标系中
        own_velocity: 自身速度
        own_heading: 自身航向角
        
        返回:
        动作 [角度调整, 速度调整]，范围均为[-1, 1]
        """
        # 如果没有目标位置，则保持当前状态
        if target_position is None:
            return [0.0, 0.0]

        # 计算到目标的方向向量
        # 在拒止区域内，这些坐标是局部坐标（相对于当前无人机位置）
        # 在拒止区域外，这些坐标是世界坐标
        dx = target_position[0] - own_position[0]
        dy = target_position[1] - own_position[1]
        
        # 计算到目标的角度
        target_angle = math.atan2(dy, dx)
        
        # 计算角度差
        angle_diff = target_angle - own_heading
        # 规范化角度差到[-π, π]范围
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # 计算到目标的距离
        distance = math.sqrt(dx**2 + dy**2)
        
        # 计算角度调整（归一化到[-1, 1]）
        # 将角度差从[-π, π]映射到[-1, 1]
        # 添加平滑因子0.7，使动作更加平滑
        angle_action = max(-1.0, min(1.0, angle_diff / math.pi)) * 0.7
        
        # 计算速度调整
        # 根据距离调整目标速度，距离越远速度越快
        # 最小速度2.0，最大速度6.0
        target_speed = min(6.0, max(2.0, distance * 0.05))  # 调整速度增益因子
        # 计算速度差并归一化到[-1, 1]
        # 速度范围[2, 6]映射到[-1, 1]
        speed_action = (target_speed - own_velocity) / 4.0  
        # 添加平滑因子0.7，避免动作过于剧烈
        speed_action = max(-1.0, min(1.0, speed_action)) * 0.7
        
        return [angle_action, speed_action]

    def learn(self):
        """
        从经验回放中学习更新网络参数 - 优化版
        简化状态处理、移除冗余操作并提高计算效率
        """
        # 检查经验回放缓冲区是否已准备好进行采样
        if not self.memory.ready():
            return

        # 从缓冲区采样一个批次的经验
        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        
        # 预处理状态数据，确保维度一致
        # 只在必要时进行填充或截断，避免重复操作
        processed_states = []
        processed_next_states = []
        
        for state in states:
            if len(state) == 48:
                processed_states.append(state)
            elif len(state) < 48:
                padded_state = np.pad(state, (0, 48 - len(state)), 'constant')
                processed_states.append(padded_state)
            else:
                processed_states.append(state[:48])
                
        for state_ in states_:
            if len(state_) == 48:
                processed_next_states.append(state_)
            elif len(state_) < 48:
                padded_state = np.pad(state_, (0, 48 - len(state_)), 'constant')
                processed_next_states.append(padded_state)
            else:
                processed_next_states.append(state_[:48])
        
        # 转换为numpy数组
        states_array = np.array(processed_states)
        next_states_array = np.array(processed_next_states)

        # 将numpy数组转换为PyTorch张量并移至指定设备
        states_tensor = T.tensor(states_array, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states_array, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        # 识别拒止区域内的状态并调整奖励权重 - 精确判断48维状态
        if states_tensor.shape[1] == 48:  # 精确判断状态维度为48
            in_denied_area = states_tensor[:, 47] > 0.5  # 拒止区域状态标记在第47维（从0开始计数）
            reward_weights = T.where(in_denied_area, 1.5, 1.0)

            # 一次完成维度调整和权重应用
            if rewards_tensor.dim() == 1:
                rewards_tensor = rewards_tensor.unsqueeze(1) * reward_weights.unsqueeze(1)
            else:
                rewards_tensor = rewards_tensor * reward_weights

        # 更新Critic网络
        with T.no_grad():
            # 从目标策略中采样下一动作
            next_action, next_log_pi, _ = self.actor.sample(next_states_tensor)
            # 使用目标网络计算下一状态的Q值
            q1_next = self.target_critic1.forward(next_states_tensor, next_action)
            q2_next = self.target_critic2.forward(next_states_tensor, next_action)
            # 取两个Q网络的最小值
            min_q_next = T.min(q1_next, q2_next)
            # 计算目标Q值 - 修复形状不匹配问题
            # 确保所有张量形状一致
            rewards_tensor = rewards_tensor.view(-1, 1)
            terminals_tensor = terminals_tensor.view(-1, 1)
            next_log_pi = next_log_pi.view(-1, 1)

            q_target = rewards_tensor + \
                       self.gamma * (1 - terminals_tensor.float()) * \
                       (min_q_next - self.alpha * next_log_pi)

        # 计算当前Q值
        q1 = self.critic1.forward(states_tensor, actions_tensor)
        q2 = self.critic2.forward(states_tensor, actions_tensor)

        # 计算Critic损失
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        critic_loss = critic1_loss + critic2_loss

        # 更新Critic网络
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        # 更新Actor网络 - 优化版
        new_action, log_prob, _ = self.actor.sample(states_tensor)
        q1_new = self.critic1.forward(states_tensor, new_action)
        q2_new = self.critic2.forward(states_tensor, new_action)
        min_q_new = T.min(q1_new, q2_new)

        # 为拒止区域内的无人机调整策略损失 - 简化版
        if 'in_denied_area' in locals():
            actor_weights = T.where(in_denied_area, 1.2, 1.0).unsqueeze(1) if log_prob.dim() > 1 else T.where(
                in_denied_area, 1.2, 1.0)
            actor_loss = (self.alpha * log_prob * actor_weights - min_q_new * actor_weights).mean()
        else:
            # 使用标准损失计算
            actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # 更新Alpha参数（熵调节系数）
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 更新目标网络参数，使用软更新方式
        self.update_network_parameters()

    def save_models(self, episode):
        """保存所有网络模型 - 高性能版"""
        # 预创建所有必要的目录（一次性操作）
        dirs = ['Actor', 'Target_critic1', 'Target_critic2', 'Critic1', 'Critic2']
        for d in dirs:
            os.makedirs(os.path.join(self.checkpoint_dir, d), exist_ok=True)

        # 高效模型保存，减少字符串操作和异常处理开销
        models_to_save = [
            (self.actor, 'Actor/SAC_actor_'),
            (self.critic1, 'Critic1/SAC_critic1_'),
            (self.critic2, 'Critic2/SAC_critic2_'),
            (self.target_critic1, 'Target_critic1/SAC_target_critic1_'),
            (self.target_critic2, 'Target_critic2/SAC_target_critic2_')
        ]

        # 批量保存模型，减少异常捕获次数
        for model, base_path in models_to_save:
            try:
                model.save_checkpoint(f"{self.checkpoint_dir}{base_path}{episode}.pth")
            except:
                pass  # 完全静默错误

    def load_models(self, episode):
        """加载所有网络模型 - 高性能版"""
        # 高效模型加载，减少字符串操作和异常处理开销
        models_to_load = [
            (self.actor, 'Actor/SAC_actor_'),
            (self.critic1, 'Critic1/SAC_critic1_'),
            (self.critic2, 'Critic2/SAC_critic2_'),
            (self.target_critic1, 'Target_critic1/SAC_target_critic1_'),
            (self.target_critic2, 'Target_critic2/SAC_target_critic2_')
        ]

        # 批量加载模型，减少异常捕获次数
        for model, base_path in models_to_load:
            try:
                model.load_checkpoint(f"{self.checkpoint_dir}{base_path}{episode}.pth")
            except:
                pass  # 完全静默错误


# 全局变量记录追踪和到达数量
# 用于在整个训练过程中累计追踪和到达的统计信息
# zhui_num = 0   # 累计追踪到失联无人机的次数
# dao_num = 0    # 累计到达目标的次数


class SACTrainer:
    """
    SAC训练器类，负责管理整个训练和测试流程
    包括环境初始化、训练循环、测试循环、结果记录等功能
    该类封装了完整的训练和测试逻辑，用于多无人机协同搜索任务
    """

    def __init__(self, n_agents, load_checkpoint=None):
        """
        初始化SAC训练器 - 高性能版
        """
        # 核心参数快速初始化
        self._n_agents = n_agents
        self._obs_dim = 48  # 48(基础状态，包含拒止区域标记) = 48
        self._action_dim = 2

        # 优化SAC算法初始化，减少参数传递
        sac_params = {
            'alpha': LR_ACTOR,
            'beta': LR_CRITIC,
            'state_dim': self._obs_dim,
            'action_dim': 2,
            'actor_fc1_dim': 256,     # 增加网络容量
            'actor_fc2_dim': 128,     # 增加网络容量
            'critic_fc1_dim': 256,    # 增加网络容量
            'critic_fc2_dim': 128,    # 增加网络容量
            'ckpt_dir': args.ckpt_dir,
            'gamma': GAMMA,
            'tau': TAU,
            'max_size': 1000000,      # 增加经验回放缓冲区大小
            'batch_size': 512         # 增加批处理大小以更好地利用GPU
        }
        self._agent = SAC(**sac_params)

        # 初始化环境和训练控制变量
        self._env = EnvWrapper(n_agents)
        # 设置环境为训练模式
        self._env._env.is_testing = False
        self._now_ep = 0
        self._step = 0
        self.best_reward = -float('inf')

        # 仅保留必要的统计变量
        self.sumway = [0.0] * n_agents
        self.endstep = [0.0] * n_agents
        self.rew = [0.0] * n_agents
        self.End = [0.0] * n_agents

        # 追踪和到达统计变量
        self.zhui_num = 0
        self.dao_num = 0

        # 失联无人机状态跟踪（保留核心功能）
        self.lostend = 0
        self.lostway = 0
        self.losttime = 0

        # 初始化拒止区域处理器
        self.denied_area_handler = DeniedAreaHandler(max_uavs=n_agents)

        # 可选的TensorBoard记录 - 可以根据需要禁用以提高性能
        # 安全检查args是否有use_tensorboard属性，避免AttributeError
        use_tensorboard = getattr(args, 'use_tensorboard', False)
        if use_tensorboard:
            self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs/trainer')
        else:
            self._sw = None

        # 检查点加载逻辑优化
        checkpoint_to_load = load_checkpoint if load_checkpoint is not None else args.checkpoint
        self._resumed_from_checkpoint = False
        if checkpoint_to_load is not None:
            self.load_checkpoint(checkpoint_to_load)
            self._resumed_from_checkpoint = True

    def _sample_global(self):
        """设置全局中心视角"""
        self._env.set_global_center(True)

    def save_checkpoint(self, episode):
        """保存训练检查点，包括模型和训练状态"""
        # 保存模型
        self._agent.save_models(episode)

        # 保存训练状态
        checkpoint_data = {
            'episode': self._now_ep,  # 当前episode数
            'step': self._step,  # 当前总步数
            'best_reward': self.best_reward,  # 历史最佳奖励
            'sumway': self.sumway,  # 各无人机总路径长度
            'endstep': self.endstep,  # 各无人机结束步数
            'rew': self.rew,  # 各无人机奖励
            'zhui_num': self.zhui_num,  # 追踪到失联无人机的次数
            'dao_num': self.dao_num,  # 到达目标的次数
            'lostend': self.lostend,  # 失联无人机是否完成任务
            'lostway': self.lostway,  # 失联无人机路径长度
            'losttime': self.losttime  # 失联无人机完成任务时间
        }

        # 每1000次episode保存经验回放缓冲区
        if self._now_ep % 1000 == 0:
            checkpoint_data['replay_buffer'] = self._agent.memory.get_state()
            print(f"经验回放缓冲区已添加到检查点中 (episode {self._now_ep})")

        checkpoint_path = os.path.join(self._agent.checkpoint_dir, f'training_state_{episode}.pth')
        # 使用weights_only=False保存检查点以确保兼容性
        try:
            T.save(checkpoint_data, checkpoint_path)
            print(f'训练状态已在episode {episode} 保存')
        except MemoryError:
            print(f'保存检查点时发生内存错误，跳过保存经验回放缓冲区')

    def load_checkpoint(self, episode):
        """加载训练检查点，包括模型和训练状态"""
        try:
            # 加载模型
            self._agent.load_models(episode)

            # 加载训练状态
            checkpoint_path = os.path.join(self._agent.checkpoint_dir, f'training_state_{episode}.pth')
            if os.path.exists(checkpoint_path):
                # 使用weights_only=False加载检查点以确保兼容性
                checkpoint_data = T.load(checkpoint_path, weights_only=False)
                self._now_ep = checkpoint_data.get('episode', 0)
                self._step = checkpoint_data.get('step', 0)
                self.best_reward = checkpoint_data.get('best_reward', -float('inf'))
                self.sumway = checkpoint_data.get('sumway', [0.0] * self._n_agents)
                self.endstep = checkpoint_data.get('endstep', [0.0] * self._n_agents)
                self.rew = checkpoint_data.get('rew', [0.0] * self._n_agents)
                self.zhui_num = checkpoint_data.get('zhui_num', 0)
                self.dao_num = checkpoint_data.get('dao_num', 0)
                self.lostend = checkpoint_data.get('lostend', 0)
                self.lostway = checkpoint_data.get('lostway', 0)
                self.losttime = checkpoint_data.get('losttime', 0)

                # 加载经验回放缓冲区状态
                if 'replay_buffer' in checkpoint_data:
                    self._agent.memory.load_state(checkpoint_data['replay_buffer'])
                    print(f'经验回放缓冲区已加载，包含 {self._agent.memory.n_samples()} 个样本')

                print(f'训练状态已在episode {episode} 加载')
                print(f'当前episode: {self._now_ep}, 当前步数: {self._step}')
            else:
                print(f'训练状态文件不存在: {checkpoint_path}')
        except Exception as e:
            print(f'加载检查点时出错: {e}')

    def train_one_episode(self):
        """
        训练一个episode（一轮完整的环境交互过程）
        在一个episode中，智能体与环境进行完整的一轮交互直到结束
        该方法实现了SAC算法的训练流程，包括动作选择、环境交互、经验存储和网络更新

        训练过程包括：
        1. 环境初始化和参数设置
        2. 与环境交互收集经验
        3. 存储经验到回放缓冲区
        4. 定期从回放缓冲区学习更新网络参数
        """
        # 根据命令行参数设置视角模式
        # GLOBAL模式：使用全局中心视角，能够观察到所有无人机的全局信息
        if args.global_ == 'GLOBAL':
            self._env.set_global_center(True)
        # LOCAL模式：使用局部视角，只能观察到局部信息
        elif args.global_ == 'LOCAL':
            self._env.set_global_center(False)
        # ANNEAL模式：使用退火策略动态调整视角
        elif args.global_ == 'ANNEAL':
            self._sample_global()
        # 如果参数不匹配任何模式，则断言失败
        else:
            assert False

        # 当前episode计数器增加1
        # 如果是从检查点恢复，则第一次不需要增加计数，因为已经设置好了
        if not (hasattr(self, '_resumed_from_checkpoint') and self._resumed_from_checkpoint):
            self._now_ep += 1
        else:
            # 清除标记，确保之后的episode正常计数
            self._resumed_from_checkpoint = False

        # 快速初始化训练变量
        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]
        states = self._env.reset()
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        # 仅在必要时获取环境信息
        target_pos = self._env._env._cpp_env.getTarget()
        obs = self._env._env._cpp_env.getObstacles()

        # 优化训练循环
        step_count = 0
        while not real_done(done):
            step_count += 1
            actions = {}
            in_states = []

            # 快速构建状态列表 - 移除冗余注释
            for seq in enum_seq:
                in_states.append(states[seq])

            # 批量获取无人机信息，减少环境调用
            uavs = self._env._env._cpp_env.getUavs()
            w = self._env._env._cpp_env.getuavW()
            v = self._env._env._cpp_env.getuavV()
            
            # 确保uavs、w和v的长度一致，过滤掉坠毁的无人机
            valid_uav_ids = list(uavs.keys())
            valid_uavs = {i: uavs[i] for i in valid_uav_ids}
            valid_w = [w[i] for i in valid_uav_ids]
            valid_v = [v[i] for i in valid_uav_ids]

            # 获取拒止区域状态
            uav_denied_status = {}
            for i in range(self._n_agents):
                uav_denied_status[i] = self._env._env.uav_denied_status.get(i, False)
            
            # 获取点云数据
            uav_pointclouds = {}
            for i in range(self._n_agents):
                uav_pointclouds[i] = self._env._env.uav_pointclouds.get(i, None)

            # 优化动作选择和分配
            out_actions = self._agent.choose_action(
                in_states, valid_uavs, obs, self._now_ep, target_pos, valid_w, valid_v,
                self._n_agents, True, uav_denied_status, uav_pointclouds, self.denied_area_handler
            )
            
            # 向量化动作分配
            for i, seq in enumerate(enum_seq):
                # 确保out_actions长度与enum_seq一致
                if i < len(out_actions):
                    actions[seq] = out_actions[i]
                else:
                    actions[seq] = [0.0, 0.0]

            # 优化存活无人机选择逻辑
            die = self._env._env._cpp_env.getStatus()
            choices = [i for i in range(self._n_agents) if not die[i]]  # 使用列表推导式

            # 执行动作并处理结果
            next_states, rewards, done, info = self._env.step(actions)
            self._step += 1

            # 优化经验存储，仅在有存活无人机时执行
            # 只存储拒止区域外的无人机经验
            for i, seq in enumerate(enum_seq):
                if not die[i] and not uav_denied_status.get(i, False):
                    # 直接存储经验，减少中间变量
                    self._agent.memory.store_transition(
                        states[seq], actions[seq], rewards[seq],
                        next_states[seq], done[seq]
                    )

            # 提高学习频率以更好地利用GPU
            # 每5步进行一次学习更新，提高GPU利用率
            if self._step % 5 == 0 and self._agent.memory.n_samples() > START_UPD_SAMPLES:
                # 从经验回放缓冲区中采样并更新网络参数
                self._agent.learn()

            # 累计奖励，用于后续的统计和分析
            for seq in enum_seq:
                total_rew[seq] += rewards[seq]

            # 更新状态，为下一步交互做准备
            states = next_states

        # 返回总奖励
        return total_rew


    def test_one_episode(self):
        """
        轻量级测试函数 - 仅保留核心评估功能以加速训练
        """
        # 快速初始化 - 移除所有打印和绘图功能
        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]

        # 重置最小化的测试变量
        self.sumway = [0.0] * self._n_agents  # 仅保留必要的统计
        self.endstep = [0.0] * self._n_agents

        # 重置环境，获取初始状态
        states = self._env.reset()
        # 设置环境为测试模式
        self._env._env.is_testing = True
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        # 最小化环境信息获取
        target_pos = self._env._env._cpp_env.getTarget()
        obs = self._env._env._cpp_env.getObstacles()

        # 简化测试循环，移除所有绘图、调试和复杂逻辑
        max_steps = 500  # 增加最大步数以允许无人机有足够时间到达目标区域
        for num in range(max_steps):
            if real_done(done):
                break

            actions = {}
            in_states = []

            # 简化状态处理
            for seq in enum_seq:
                in_states.append(states[seq])

            # 批量获取无人机信息
            uavs = self._env._env._cpp_env.getUavs()
            w = self._env._env._cpp_env.getuavW()
            v = self._env._env._cpp_env.getuavV()
            
            # 确保uavs、w和v的长度一致，过滤掉坠毁的无人机
            valid_uav_ids = list(uavs.keys())
            valid_uavs = {i: uavs[i] for i in valid_uav_ids}
            valid_w = [w[i] for i in valid_uav_ids]
            valid_v = [v[i] for i in valid_uav_ids]
            
            # 获取目标点位置
            target_pos = self._env._env._cpp_env.getTarget()
            
            # 获取障碍物信息
            obs = self._env._env._cpp_env.getObstacles()
            
            # 获取拒止区域状态
            uav_denied_status = {}
            for i in range(self._n_agents):
                uav_denied_status[i] = self._env._env.uav_denied_status.get(i, False)
            
            # 获取点云数据
            uav_pointclouds = {}
            for i in range(self._n_agents):
                uav_pointclouds[i] = self._env._env.uav_pointclouds.get(i, None)

            # 简化动作选择
            out_actions = self._agent.choose_action(
                in_states, valid_uavs, obs, self._now_ep, target_pos, valid_w, valid_v,
                self._n_agents, False, uav_denied_status, uav_pointclouds, self.denied_area_handler
            )  # 评估模式

            # 分配动作
            for i, seq in enumerate(enum_seq):
                # 确保out_actions长度与enum_seq一致
                if i < len(out_actions):
                    actions[seq] = out_actions[i]
                else:
                    actions[seq] = [0.0, 0.0]

            # 执行动作并更新
            next_states, rewards, done, info = self._env.step(actions)

            # 仅更新必要的统计
            for seq in enum_seq:
                total_rew[seq] += rewards[seq] / 100  # 保持与原有奖励缩放一致

            states = next_states

        # 可视化测试结果
        self._visualize_test_episode()

        # 优化最佳模型保存逻辑，减少不必要的计算
        avg_reward = sum(total_rew.values()) / len(total_rew)
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            # 仅在较大的episode间隔才保存最佳模型，减少I/O操作
            if self._now_ep >= 500 or self.best_reward > -100:  # 设置合理的阈值
                self.save_checkpoint('best')

        # 恢复环境为训练模式
        self._env._env.is_testing = False
        # 返回总奖励
        return total_rew

    def _visualize_test_episode(self):
        """
        可视化测试episode结果
        绘制无人机路径、障碍物和拒止区域
        """
        try:
            import draw as d
            
            # 绘制背景
            d.drawback()
            
            # 绘制目标点
            target_pos = self._env._env._cpp_env.getTarget()
            d.drawtarget(target_pos.x, target_pos.y)
            
            # 绘制障碍物
            obs = self._env._env._cpp_env.getObstacles()
            # 只绘制真正的障碍物（索引0和1），避免将拒止区域误作为障碍物绘制
            for i in range(2):
                d.drawobs(obs[i].x, obs[i].y, i)
            
            # 绘制拒止区域
            from env1 import DENIED_AREA_RADIUS
            # 根据用户反馈，拒止区域坐标应为(200,200)和(350,320)
            d.draw_denied_area(200, 200, DENIED_AREA_RADIUS, color='red', alpha=0.2)
            d.draw_denied_area(350, 320, DENIED_AREA_RADIUS, color='red', alpha=0.2)
            
            # 绘制无人机最终位置和路径
            final_uavs = self._env._env._cpp_env.getUavs()
            uav_positions = list(final_uavs.values())
            
            # 绘制无人机路径（如果可用）
            # 从环境获取无人机历史位置
            try:
                uav_history = self._env._env._cpp_env.getUavPositionsHistory()
                # 传递uav_ids参数以确保图例正确显示
                uav_ids = list(uav_history.keys()) if uav_history else None
                d.draw_uav_paths_with_colors(uav_history, uav_ids)
            except Exception as e:
                # 如果无法获取历史位置，则只绘制最终位置
                print(f"绘制无人机路径时出错: {e}")
                pass
            
            # 绘制无人机最终位置
            # 按照uav_id排序，确保颜色和编号的对应关系一致
            sorted_uav_items = sorted(final_uavs.items())
            for i, (uav_id, pos) in enumerate(sorted_uav_items):
                color = d.UAV_COLORS[i % len(d.UAV_COLORS)]
                d.draw_uav(pos[0], pos[1], color)
            
            # 保存图片
            d.savepic(self._now_ep)
            print(f"测试结果图片已保存: sac1_results/pic-{self._now_ep}.png")
        except Exception as e:
            print(f"绘图时出错: {e}")
