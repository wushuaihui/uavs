# 导入数学库
import math as m
# 导入随机数生成库
import random
# 导入数值计算库
import numpy as np
# 导入绘图库
from matplotlib import pyplot as plt

# 定义二维点类
class Point2D:
    def __init__(self):
        # 初始化x坐标（复数类型）
        self.x = complex()
        # 初始化y坐标（复数类型）
        self.y = complex()

# 定义控制信息类
class ControlInfo:
    def __init__(self):
        # 初始化控制参数数组（长度2）
        self.c = [None]*2

# 定义观测类
class Observation:
    def __init__(self):
        # 初始化观测数组（长度48）
        self.o= [None] * 48

# 定义键值对类
class pair:
    def __init__(self):
        # 初始化第一个值
        self.first = 0
        # 初始化第二个值
        self.second = 0

# 计算二维欧氏距离
def l2norm (x,y):
    # 返回x和y的平方和开方
    return m.sqrt(x * x + y * y)

# Sigmoid激活函数
def sigmoid(x):
    # 计算Sigmoid函数值
    y=1/(1+np.exp(-x))
    # 计算Sigmoid导数
    dy=y*(1-y)
    # 返回导数
    return dy

# 根据值查找字典键
def getDictKey_1(myDict, value):
     # 遍历字典项
     for k, v in myDict.items():
         # 找到匹配值则退出循环
         # 使用近似比较处理浮点数精度问题
         if abs(v - value) < 1e-6:
             return k
     # 如果没有找到匹配项，返回None
     return None

# 无人机坠毁惩罚值
DIE_PENALTY = 100
# 无人机碰撞半径
UAV_COLLISION_R = 3
# 圆周率常量
PI=3.14159265358979323846
# 无人机间通信距离
UAV_UAV_COMM = 50
# 障碍物字典
m_obstacless = {}
# 障碍物数量
OBSTACLE_CNT = 0
# 目标点
m_target = Point2D()
# 无人机坠毁状态列表
m_die = []
# 无人机状态列表
m_status = []
# 进入目标区域的无人机数量
in_target_area = 0
# 与障碍物碰撞次数
collision_with_obs = 0
# 无人机间碰撞次数
collision_with_uav = 0
# 上一时刻位置字典
m_prev_pos = {}
# 下一时刻位置字典
m_next_pos = {}
# 障碍物列表
m_obstacles = []
# 碰撞记录列表
m_collision = []
# 本步坠毁状态列表
die_on_this_step = []
# 最大距离惩罚值
MAX_DIST_PENALTY = 50
# 最大有效距离范围
MAX_DIST_RANGE = 500
# 目标区域半径
TARGET_R = 50
# 任务完成奖励
TASK_REW = 100

# 获取无人机状态
def getStatus(self):
    # 返回坠毁状态列表
    return m_die

# 二维无人机模型类
class UAVModel2D:
    # 初始化无人机
    def __init__(self, x, y, v, w):
        # x坐标
        self.m_x = x
        # y坐标
        self.m_y = y
        # 速度
        self.m_v = v
        # 航向角
        self.m_w = w

    # 单步移动
    def step(self, ang, acc):
        # 更新航向角
        self.m_w += ang
        # 更新速度
        self.m_v += acc
        # 速度上限约束（6）
        if (self.m_v > 6) :
            self.m_v = 6
        # 速度下限约束（2）
        if (self.m_v < 2) :
            self.m_v = 2
        # 航向角上限约束（3π/4）
        while (self.m_w > 3*PI/4) :
            self.m_w = 3*PI/4
        # 航向角下限约束（-π/4）
        while (self.m_w < -PI/4) :
            self.m_w = 0
        # 更新x坐标（速度*cos(航向角)）
        self.m_x += self.m_v * m.cos(self.m_w)
        # 更新y坐标（速度*sin(航向角)）
        self.m_y += self.m_v * m.sin(self.m_w)

# 无人机字典
m_uavs = {}
# 起始位置字典
start = {}

# 多无人机环境类
class ManyUavEnv:
    # 初始化环境
    def __init__(self, uav_cnt, random_seed,uav_die = True):
        # 是否允许无人机坠毁
        self.uav_die = uav_die
        # 无人机数量
        self.m_uav_cnt = uav_cnt
        # 随机种子
        self.m_rnd_engine = random_seed
        # 目标点字典
        self.m_target = {}
        # 步数计数器
        self.m_steps = 0
        # 成功计数
        self.succ_cnt = 0
        # 总路径长度列表
        self.sumway = [0.0]*uav_cnt
        # 结束步数列表
        self.endstep = [0.0]*uav_cnt

    # 获取状态
    def getStatus(self):
        # 返回坠毁状态列表
        return m_die

    # 重置环境
    def reset(self):
        # 清空无人机字典
        m_uavs.clear()
        # 清空障碍物列表
        m_obstacles.clear()
        # 初始化计数器
        k=0
        # 初始y坐标
        a = 49
        # 初始化所有无人机
        for i in range (self.m_uav_cnt) :
            # 每3架重置分组
            if(k == 3):
                k = 0
            # 每3架调整y坐标
            if(i % 3 ==0):
                a = a - 3
            # 增加组内计数
            k = k + 1
            # 创建无人机实例
            m_uavs[i] = UAVModel2D(9+3*k,a , 2.0, PI/4)
            # 记录起始位置
            start[i] = Point2D()
            start[i] .x = m_uavs[i].m_x
            start[i] .y = m_uavs[i].m_y

        # 设置障碍物0位置
        m_obstacless[3] = Point2D()
        m_obstacless[3].x= 200
        m_obstacless[3].y = 200
        # 设置障碍物1位置
        m_obstacless[2] = Point2D()
        m_obstacless[2].x = 350
        m_obstacless[2].y = 320
        # 设置障碍物2位置
        m_obstacless[1] = Point2D()
        m_obstacless[1].x = 200
        m_obstacless[1].y = 320
        # 设置障碍物3位置
        m_obstacless[0] = Point2D()
        m_obstacless[0].x = 300
        m_obstacless[0].y = 100
        # 设置目标点位置
        m_target.x = 475
        m_target.y = 475
        # 重置步数
        m_steps = 0
        # 清空坠毁列表
        m_die.clear()
        # 清空状态列表
        m_status.clear()
        # 初始化所有无人机状态
        for i in range (self.m_uav_cnt):
            # 初始未坠毁
            m_die.append(False)
            # 初始状态0
            m_status.append(0)
        # 重置成功计数
        self.succ_cnt = 0

    # 环境单步执行
    def step(self,control):
        # 重置目标区域计数
        self.in_target_area = 0
        # 重置无人机碰撞计数
        self.collision_with_uav = 0
        # 清空上一位置
        m_prev_pos.clear()
        # 清空下一位置
        m_next_pos.clear()
        # 遍历所有无人机
        for i in range (self.m_uav_cnt):
            # 跳过已坠毁无人机
            if (m_die[i]) :continue
            # 记录当前位置
            m_prev_pos[i] = [m_uavs[i].m_x, m_uavs[i].m_y]
            # 执行控制指令（角度变化，加速度）
            m_uavs[i].step(control[i][0], control[i][1])
            # 记录新位置
            m_next_pos[i] = [m_uavs[i].m_x, m_uavs[i].m_y]
            # 启发式修改点（可调整速度和角速度）

        # 增加步数计数
        self.m_steps = self.m_steps + 1
        # 步数循环（200步重置）
        if self.m_steps == 200:
            self.m_steps = self.m_steps - 200

    # 获取观测值
    def getObservations(self):
        # 初始化结果列表
        result = []
        # 初始化中心点
        center = Point2D()
        center.x = 0
        center.y = 0
        # 有效无人机计数
        cnt = 0
        # 计算所有无人机中心位置
        for i in range(self.m_uav_cnt):
            # 跳过已完成任务无人机
            if m_status[i] != 1:
                # 累加x坐标
                center.x += m_uavs[i].m_x
                # 累加y坐标
                center.y += m_uavs[i].m_y
                # 增加计数
                cnt += 1
        # 计算平均中心位置
        if cnt != 0:
            center.x /= cnt
            center.y /= cnt
        # 为每个无人机生成观测
        for i in range(self.m_uav_cnt):
            # 初始化观测数组
            obs = Observation().o
            # 归一化x坐标（0-1）
            obs[0] = (m_uavs[i].m_x) / 500.
            # 归一化y坐标（0-1）
            obs[1] = (m_uavs[i].m_y) / 500.
            # 归一化速度（0-1）
            obs[2] = (m_uavs[i].m_v - 2) / 4.
            # 归一化航向角（0-1）
            obs[3] = (m_uavs[i].m_w+PI/4) / (PI)
            # 初始化距离字典
            index_dist = {}
            # 计算与其他无人机的距离
            for j in range (self.m_uav_cnt):
                # 跳过自身
                if (j == i) : continue
                # 跳过坠毁无人机
                if(m_die[j]): continue
                # 确保无人机j在m_uavs字典中
                if j not in m_uavs: continue
                # 计算欧氏距离
                dist = l2norm((m_uavs[j].m_x - m_uavs[i].m_x), (m_uavs[j].m_y - m_uavs[i].m_y ))
                # 存储距离信息
                index_dist[j] = dist
            # 按距离排序（升序）
            index_dist = sorted(index_dist.items(),  key=lambda d: d[1], reverse=False)
            # 转换为字典
            index_dist = dict(index_dist)#字典
            # 获取距离值
            value = sorted(index_dist.values())#value
            # 处理最近的8架无人机
            for j in range (8):
                # 检查是否在通信范围内
                if j < len(index_dist) and value[j] < UAV_UAV_COMM:
                    # 获取无人机ID
                    key = getDictKey_1(index_dist, value[j])
                    # 确保找到了对应的键且键在m_uavs字典中
                    if key is not None and key in m_uavs:
                        # 归一化邻居x坐标
                        obs[4 + j * 4] = (m_uavs[key].m_x) / 500.
                        # 归一化邻居y坐标
                        obs[5 + j * 4] = (m_uavs[key].m_y) / 500.
                        # 归一化邻居速度
                        obs[6 + j * 4] = (m_uavs[key].m_v - 2) / 4.
                        # 归一化邻居航向角
                        obs[7 + j * 4] = (m_uavs[key].m_w+PI/4) / (PI)
                    else:
                        # 如果未找到键或键不在m_uavs中，填充默认值
                        obs[4 + j * 4] = -1.0
                        obs[5 + j * 4] = -1.0
                        obs[6 + j * 4] = 0.0
                        obs[7 + j * 4] = 0.0
                else:
                    # 超出范围填充默认值
                    obs[4 + j * 4] = -1.0
                    obs[5 + j * 4] = -1.0
                    obs[6 + j * 4] = 0.0
                    obs[7 + j * 4] = 0.0
            # 归一化障碍物x坐标
            for k in range (4):
                obs[36+k] = (m_obstacless[k].x ) / 500
            # 归一化障碍物y坐标
            for k in range (4):
                obs[40+k] = (m_obstacless[k].y) / 500
            # 目标点相对位置x
            obs[44] = (m_target.x - 450.) / 50.
            # 目标点相对位置y
            obs[45] = (m_target.y - 450.) / 50.
            # 坠毁状态（0/1）
            obs[46] = int(m_die[i])
            # 归一化步数（0-1）
            obs[47] = self.m_steps / 200.
            # 添加到结果列表
            result.append(obs)
        # 返回观测数组
        return np.array(result)

    # 计算奖励值
    def getRewards(self):
        # 声明全局碰撞标志
        global collisiono
        # 初始化奖励数组
        result = [0.0]*self.m_uav_cnt
        # 遍历所有无人机
        for i in range (self.m_uav_cnt):
            # 跳过已坠毁无人机
            if(m_die[i]):
                result[i] = 0
                continue
            # 重置路径长度
            self.sumway[i] = 0
            # 基础惩罚
            result[i] = result[i] - 2
            # 计算到目标的距离变化
            dp = l2norm(m_prev_pos[i][0] - m_target.x, m_prev_pos[i][1] - m_target.y)
            dn = l2norm(m_next_pos[i][0] - m_target.x, m_next_pos[i][1] - m_target.y)
            # 计算移动距离
            dstep = l2norm(m_next_pos[i][0]-m_prev_pos[i][0],m_next_pos[i][1]-m_prev_pos[i][1])
            # 奖励距离缩短
            result[i] = 1.1*(dp - dn) +result[i]
            # 记录路径长度
            self.sumway[i] = dstep
            # 初始化本步坠毁状态
            die_on_this_step =[False]*self.m_uav_cnt
            # 检查无人机间碰撞
            for j in range(self.m_uav_cnt):
                # 跳过自身和已坠毁
                if(m_die[j] or i == j):continue
                # 检查碰撞距离
                if(l2norm(m_next_pos[j][0] - m_next_pos[i][0], m_next_pos[j][1] - m_next_pos[i][1]) < 3):
                    # 碰撞惩罚
                    result[i] -= 15
                    break
            # 检查障碍物碰撞
            collisiono = False
            for k in range (4):
                # 检查与障碍物距离
                if(l2norm(m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) < 30+k*5):
                    collisiono = True
                    break
            # 障碍物碰撞处理
            if(collisiono):
                # 碰撞惩罚
                result[i] -= 75
                # 标记本步坠毁
                die_on_this_step[i] = True
            # 检查接近障碍物
            for k in range (4):
                if(k>1):
                    # 检查危险距离
                    if(l2norm(m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) > 30+k*5 and l2norm(m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) < 40+k*5):
                        # 接近惩罚
                        result[i] -= 2
            # 集群奖励
            for j in range (self.m_uav_cnt):
                # 跳过已坠毁
                if(m_die[j]): continue
                if i != j :
                    # 计算无人机间距离
                    dist = l2norm(m_next_pos[j][0]- m_next_pos[i][0], m_next_pos[j][1] - m_next_pos[i][1])
                    # 在合理距离内给予奖励
                    if (dist < 20 and dist > 3) :
                        result[i] += 0.7
            # 到达目标区域
            if (l2norm(m_next_pos[i][0] - m_target.x, m_next_pos[i][1] - m_target.y) < 20) :
                # 到达奖励
                result[i] += 150
                # 记录结束步数
                if (die_on_this_step[i] == False):
                    self.endstep[i] = self.m_steps
                # 标记本步坠毁
                die_on_this_step[i] = True
        # 更新坠毁状态
        for i in range (self.m_uav_cnt) :
            if die_on_this_step[i]:
                m_die[i] = True
        # 返回奖励数组
        return result

    # 检查是否结束
    def isDone(self):
        all_die = True
        # 检查所有无人机状态
        for i in range (self.m_uav_cnt) :
            if (m_die[i] == False) :all_die = False
        # 结束条件：达到最大步数或全部坠毁
        return (self.m_steps % 200 == 0) or all_die

    # 获取障碍物位置
    def getObstacles(self):
        return m_obstacless

    # 获取无人机位置
    def getUavs(self):
        return m_next_pos

    # 获取碰撞记录
    def getCollision(self):
        return m_collision

    # 获取目标点
    def getTarget(self):
        return m_target

    # 获取路径长度
    def getWay(self):
        return self.sumway

    # 获取结束步数
    def getEnd(self):
        return self.endstep

    # 获取航向角
    def getuavW(self):
        w = [0.0]*self.m_uav_cnt
        for i in range (self.m_uav_cnt):
            w[i] = m_uavs[i].m_w
        return w
    
    # 获取速度
    def getuavV(self):
        v = [0.0] * self.m_uav_cnt
        for i in range(self.m_uav_cnt):
            v[i] = m_uavs[i].m_v
        return v