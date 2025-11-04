import numpy as np
import env1

class ManyUavEnv:
    """
    多无人机环境类，作为Python层与C++环境之间的接口
    """

    TARGET_R = 100      # 目标半径
    COLLISION_R = 30    # 碰撞半径

    def __init__(self, uav_cnt, seed=None):
        """
        初始化多无人机环境
        :param uav_cnt: 无人机数量
        :param seed: 随机种子，用于环境初始化的随机性控制
        """
        # 直接使用Python实现的环境
        self._cpp_env = env1.ManyUavEnv(uav_cnt)
        self._viewer = None  # 可视化查看器，初始为空
    
    # 添加属性访问器，以便外部代码可以访问_cpp_env中的属性
    @property
    def uav_denied_status(self):
        """返回拒止区域状态字典"""
        return self._cpp_env.uav_denied_status
    
    @property
    def uav_pointclouds(self):
        """返回点云数据字典"""
        return self._cpp_env.uav_pointclouds

    def reset(self):
        """
        重置环境到初始状态
        :return: 初始观测状态
        """
        self._cpp_env.reset()
        obs = self._cpp_env.getObservations()
        return np.array(obs)

    def step(self, actions):
        """
        执行一步动作
        :param actions: 动作列表
        :return: (观测, 奖励, 是否结束, 额外信息)四元组
        """
        self._cpp_env.step(actions)
        obs = np.array(self._cpp_env.getObservations())
        rewards = np.array(self._cpp_env.getRewards())
        done = self._cpp_env.isDone()
        return obs, rewards, done, {}

class EnvWrapper:#需要包装的

    def __init__(self, n_agents):
        self._env = ManyUavEnv(n_agents, 123)
        self._n_agents = n_agents
        self._global_center = True

    def set_global_center(self, value):
        self._global_center = value

    def reset(self):
        """
        重置环境并返回初始状态
        将环境重置到初始状态，并将初始观测状态按无人机编号组织成字典格式返回
        :return: 以字典形式返回各无人机的初始观测状态，格式为 {'uav_0': state0, 'uav_1': state1, ...}
        """
        # 调用底层环境的reset方法，获取所有无人机的初始观测状态
        s = self._env.reset()

        # 初始化结果字典
        result = {}
        # 遍历所有无人机，将每个无人机的观测状态按'uav_i'的键名存入字典
        for i in range(self._n_agents):
            # 使用np.copy确保返回的是状态数组的副本，避免外部修改影响环境内部状态
            result[f'uav_{i}'] = np.copy(s[i])
        return result

    def step(self, actions):
        act = []
        for i in range(self._n_agents):
            act.append(actions[f'uav_{i}'] * np.array([np.pi / 4, 1.0]))
        s, r, done, info = self._env.step(act)

        result_s = {}
        result_r = {}
        result_d = {}
        for i in range(self._n_agents):
            result_s[f'uav_{i}'] = np.copy(s[i])
            result_r[f'uav_{i}'] = r[i]
            result_d[f'uav_{i}'] = done

        result_d['__all__'] = all(result_d.values())
        return result_s, result_r, result_d, {}

