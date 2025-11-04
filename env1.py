# 导入数学库
import math as m
# 导入随机数生成库
import random
# 导入数值计算库
import numpy as np
# 导入绘图库
from matplotlib import pyplot as plt
# 导入Shapely库用于几何计算
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

# 导入激光雷达模拟器
# from lidar09 import RockObstacleSimulator  # 注释掉不存在的模块

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
         if v == value:
             break
     # 返回对应的键
     return k

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
# 激光雷达图像维度
# 修改：将LIDAR_IMAGE_DIM设置为0，因为我们不再需要雷达图像观测
# 在拒止区域外不需要雷达观测，在拒止区域内只需要点云数据，不生成图像
LIDAR_IMAGE_DIM = 0  # 0


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
# 存储无人机历史位置
uav_positions_history = {}

# 存储统一的拒止区域信息
m_denied_areas = []  # 存储所有拒止区域信息的数组

# 存储无人机历史位置用于路径持久性计算
uav_histories = {}  # 以uav_id为键，位置历史列表为值

# 存储每个无人机的拒止区域状态
uav_denied_status = {}  # 以uav_id为键，是否在拒止区域为值

# 存储每个无人机的惯性导航估计位置
# 在拒止区域内，无人机无法获取真实位置信息，需要通过惯性导航进行位置估计
# 估计基于速度、航向角和上一时刻位置进行推算
uav_inertial_positions = {}  # 以uav_id为键，估计位置为值

# 存储每个无人机的LED状态（True表示开启，False表示关闭）
uav_led_status = {}  # 以uav_id为键，LED状态为值

# 存储点云数据用于拒止区域内的无人机处理
uav_pointclouds = {}  # 以uav_id为键，点云数据为值

class ManyUavEnv:
    # 初始化环境
    def __init__(self, n_agents: int = 5) -> None:
        """
        初始化环境
        :param n_agents: 无人机数量
        """
        # 保存无人机数量
        self.m_uav_cnt = n_agents
        # 初始化测试标志为False（默认为训练模式）
        self.is_testing = False
        # 初始化步数计数器
        self.m_steps = 0
        # 初始化路径长度数组
        self.sumway = [0.0] * n_agents
        # 初始化结束步数字典
        self.endstep = {}

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
            # 初始化位置历史记录
            if i not in uav_positions_history:
                uav_positions_history[i] = []
            # 每次重置环境时清空历史记录，避免不同episode之间的路径叠加
            uav_positions_history[i].clear()
            uav_positions_history[i].append((m_uavs[i].m_x, m_uavs[i].m_y))

        # ========== 拒止区域状态初始化 ==========
        # 清空拒止区域状态字典
        uav_denied_status.clear()
        # 清空惯性导航位置字典
        uav_inertial_positions.clear()
        # 清空LED状态字典
        uav_led_status.clear()
        # 清空点云数据字典
        uav_pointclouds.clear()
        
        # 清空本步坠毁状态列表
        die_on_this_step.clear()
        
        # 初始化所有无人机的拒止区域状态为False（不在拒止区域内）
        for i in range(self.m_uav_cnt):
            uav_denied_status[i] = False
            uav_inertial_positions[i] = (m_uavs[i].m_x, m_uavs[i].m_y)  # 初始化惯性导航位置
            uav_led_status[i] = True  # 初始化LED为开启状态
            uav_pointclouds[i] = None
            # 初始化位置历史记录
            if i not in uav_positions_history:
                uav_positions_history[i] = []
            # 每次重置环境时清空历史记录，避免不同episode之间的路径叠加
            uav_positions_history[i].clear()
            uav_positions_history[i].append((m_uavs[i].m_x, m_uavs[i].m_y))
            # 确保die_on_this_step列表长度与无人机数量一致
            if i >= len(die_on_this_step):
                die_on_this_step.append(False)
            else:
                die_on_this_step[i] = False

        # ========== 障碍物定义（物理碰撞检测用） ==========
        # 只保留真正的障碍物，移除错误分类的拒止区域
        # 设置障碍物0位置（实际障碍物，非拒止区域）
        m_obstacless[1] = Point2D()
        m_obstacless[1].x = 200
        m_obstacless[1].y = 320
        # 设置障碍物1位置（实际障碍物，非拒止区域）
        m_obstacless[0] = Point2D()
        m_obstacless[0].x = 300
        m_obstacless[0].y = 100
        
        # ========== 拒止区域定义（信号衰减和通信限制用） ==========
        # 创建拒止区域中心点
        # 根据用户反馈，拒止区域坐标应为(200,200)和(350,320)
        center_point_0 = Point2D()
        center_point_0.x = 200
        center_point_0.y = 200
        
        center_point_1 = Point2D()
        center_point_1.x = 350
        center_point_1.y = 320
        
        # 清空拒止区域列表
        m_denied_areas.clear()
        
        # 初始化反射率特征
        uav_reflectance_features = {}
        for i in range(self.m_uav_cnt):
            # 为每架无人机分配唯一的反射率特征（0.8-1.0范围）
            # 用于通过LED闪烁频率识别不同的无人机
            uav_reflectance_features[i] = 0.8 + (i * 0.025)
        
        # 创建拒止区域多边形
        # 显示拒止区域半径信息
        
        # 遍历定义的拒止区域中心点，为每个中心点创建一个圆形拒止区域
        for i, center_point in enumerate([center_point_0, center_point_1]):
            # 初始化用于存储圆形边界点的列表
            circle_points = []
            # 获取当前拒止区域中心点的坐标
            center_x = center_point.x
            center_y = center_point.y
            
            # 通过在圆周上取点来近似表示圆形拒止区域边界
            # 从0度到360度，每隔10度取一个点，共36个点
            for angle in range(0, 360, 10):
                # 将角度转换为弧度
                rad = m.radians(angle)
                # 计算圆周上点的x坐标: 中心x坐标 + 半径 * cos(角度)
                x = center_x + DENIED_AREA_RADIUS * m.cos(rad)
                # 计算圆周上点的y坐标: 中心y坐标 + 半径 * sin(角度)
                y = center_y + DENIED_AREA_RADIUS * m.sin(rad)
                # 将计算得到的点添加到边界点列表中
                circle_points.append((x, y))
            
            # 使用Shapely库创建多边形对象来表示拒止区域
            # circle_points包含了圆周上的36个点，构成一个近似的圆形
            denied_area = Polygon(circle_points)
            
            # 将创建的拒止区域添加到全局拒止区域列表中
            # 存储的信息包括：
            # - type: 拒止区域类型（CUSTOM表示自定义区域）
            # - polygon: 表示区域边界的多边形对象
            # - center: 区域中心点
            # - radius: 区域半径
            # - id: 区域唯一标识符
            m_denied_areas.append({
                'type': DENIED_AREA_TYPE['CUSTOM'],
                'polygon': denied_area,
                'center': center_point,
                'radius': DENIED_AREA_RADIUS,
                'id': i
            })
        
        # 重要说明：
        # 1. 障碍物(m_obstacless)：仅包含真正的物理障碍物，用于物理碰撞检测
        # 2. 拒止区域(m_denied_areas)：完全独立，仅用于信号衰减和通信限制
        # 3. 两者概念完全分离，使用不同的数组单独管理
        # 4. 现在代码中已正确移除了错误分类为障碍物的拒止区域
        # 5. 拒止区域和障碍物使用完全相同的实现方式，直接在各自数组中通过索引存储Point2D对象
        # 设置目标点位置
        m_target.x = 475
        m_target.y = 475
        # 重置步数
        self.m_steps = 0
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
            # 初始化拒止区域状态
            uav_denied_status[i] = False
            uav_inertial_positions[i] = (m_uavs[i].m_x, m_uavs[i].m_y)  # 初始化惯性导航位置
            uav_led_status[i] = True  # 初始化LED为开启状态
            # 初始化结束步数
            self.endstep[i] = 0
            # 确保die_on_this_step列表长度与无人机数量一致
            if i >= len(die_on_this_step):
                die_on_this_step.append(False)
            else:
                die_on_this_step[i] = False
            
        # 初始化激光雷达模拟器，使用环境中的障碍物
        # 支持更多的无人机友机检测
        # 只访问已定义的障碍物（0和1）
        env_obstacles = [(m_obstacless[i].x, m_obstacless[i].y) for i in range(2)]
        # 根据无人机数量动态设置num_drones参数，支持最多8个友机检测
        num_drones = min(self.m_uav_cnt - 1, 8) if self.m_uav_cnt > 1 else 0
        # 导入并使用EnvironmentObstacleSimulator类
        try:
            from lidar01 import EnvironmentObstacleSimulator
            # 修改：将d_max设置为400.0以匹配lidar03.py中的设置
            self.lidar_simulator = EnvironmentObstacleSimulator(num_drones=num_drones, h=3.0, d_max=90.0, env_obstacles=env_obstacles)
            # 设置无人机反射率特征
            if self.lidar_simulator is not None:
                self.lidar_simulator.uav_reflectance_features = uav_reflectance_features
                # 设置拒止区域检查器，用于判断无人机是否在拒止区域内
                # 这样点云生成器就能知道哪些无人机的LED应该关闭
                self.lidar_simulator.denied_area_checker = self.is_in_denied_area
        except ImportError:
            # 如果无法导入，则使用默认参数
            print("无法导入lidar01.py，使用默认参数")
            self.lidar_simulator = None

        # 重置成功计数
        self.succ_cnt = 0
        
        # 返回初始观测值
        return self.getObservations()

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
            # 更新无人机位置历史记录（用于路径绘制）
            if i in uav_positions_history:
                uav_positions_history[i].append((m_uavs[i].m_x, m_uavs[i].m_y))
            else:
                uav_positions_history[i] = [(m_uavs[i].m_x, m_uavs[i].m_y)]
            
            # 检查当前无人机是否在任何一个拒止区域内
            in_denied_area = False
            # 遍历所有已定义的拒止区域
            for denied_area_info in m_denied_areas:
                # 使用is_in_denied_area函数检查无人机是否在当前拒止区域内
                if is_in_denied_area((m_uavs[i].m_x, m_uavs[i].m_y), denied_area_info['polygon']):
                    in_denied_area = True
                    break
            
            # 更新无人机的拒止区域状态和LED状态
            if in_denied_area and not uav_denied_status.get(i, False):
                # 刚刚进入拒止区域
                uav_denied_status[i] = True
                uav_led_status[i] = False  # 关闭LED
            elif not in_denied_area and uav_denied_status.get(i, False):
                # 刚刚离开拒止区域
                uav_denied_status[i] = False
                uav_led_status[i] = True   # 恢复LED
            
            # 更新惯性导航估计位置
            # 在拒止区域内使用惯性导航估计，在拒止区域外使用真实位置
            # 惯性导航基于速度和航向角进行位置推算，随着时间推移会产生累积误差
            if uav_denied_status.get(i, False):
                # 在拒止区域内，使用惯性导航估计位置
                # 基于速度和航向角进行位置估计
                old_x, old_y = uav_inertial_positions[i]
                # 使用速度和航向角估计新位置
                # 这是一个简化的惯性导航模型，实际系统中还应考虑加速度、姿态等因素
                estimated_x = old_x + m_uavs[i].m_v * m.cos(m_uavs[i].m_w)
                estimated_y = old_y + m_uavs[i].m_v * m.sin(m_uavs[i].m_w)
                uav_inertial_positions[i] = (estimated_x, estimated_y)
            else:
                # 在拒止区域外，使用真实位置并同步惯性导航估计
                # 当无人机离开拒止区域时，重新校准惯性导航系统
                uav_inertial_positions[i] = (m_uavs[i].m_x, m_uavs[i].m_y)

        # 增加步数计数
        self.m_steps = self.m_steps + 1
        # 步数循环（200步重置）
        if self.m_steps == 200:
            self.m_steps = self.m_steps - 200


    # 获取观测值
    def getObservations(self):
        # 初始化结果列表，用于存储所有无人机的观测数据
        result = []
        # 初始化中心点，用于计算无人机群的中心位置
        center = Point2D()
        center.x = 0
        center.y = 0
        # 有效无人机计数，用于计算平均中心位置
        cnt = 0
        # 计算所有未完成任务的无人机中心位置
        for i in range(self.m_uav_cnt):
            # 跳过已完成任务的无人机（状态为1表示已完成）
            if m_status[i] != 1:
                # 累加x坐标
                center.x += m_uavs[i].m_x
                # 累加y坐标
                center.y += m_uavs[i].m_y
                # 增加有效无人机计数
                cnt += 1
        # 计算无人机群的平均中心位置
        if cnt != 0:
            center.x /= cnt
            center.y /= cnt
            
        # 为每个无人机生成观测数据
        # 修改：观测空间总长度为48维（移除了激光雷达图像数据）
        for i in range(self.m_uav_cnt):
            # 初始化观测数组，只包含传统观测数据，不包含激光雷达图像数据
            obs = [0.0] * 48
            
            # 检查当前无人机是否在任何一个拒止区域内
            in_denied_area = False
            # 遍历所有已定义的拒止区域
            for denied_area_info in m_denied_areas:
                # 使用is_in_denied_area函数检查无人机是否在当前拒止区域内
                if is_in_denied_area((m_uavs[i].m_x, m_uavs[i].m_y), denied_area_info['polygon']):
                    # 如果在拒止区域内，标记状态并更新无人机的拒止区域状态
                    in_denied_area = True
                    # 检查是否是刚刚进入拒止区域
                    if not uav_denied_status.get(i, False):
                        uav_denied_status[i] = True  # 标记无人机在拒止区域内
                    break
            
            # 如果无人机不在拒止区域内，但之前在，说明刚刚离开拒止区域
            if not in_denied_area and uav_denied_status.get(i, False):
            # 如果无人机不在拒止区域内，清除其拒止区域状态标记
                if not in_denied_area:
                    uav_denied_status[i] = False
            
            # 获取点云数据用于拒止区域内的无人机处理
            # 在拒止区域外不需要点云数据，在拒止区域内需要点云数据进行聚类分析
            if uav_denied_status.get(i, False):
                # 获取环境中其他无人机的点云数据，用于本机激光雷达扫描识别
                pointcloud = self.getLidarPointcloudForUAV(i)
                # 保存点云数据用于后续处理
                uav_pointclouds[i] = pointcloud
            else:
                # 在拒止区域外，清除点云数据
                uav_pointclouds[i] = None
            
            # 根据无人机是否在拒止区域内提供不同的观测信息
            # 如果无人机在拒止区域内，则无法获取完整的位置信息，但可以获得点云数据
            if uav_denied_status.get(i, False):
                # 在拒止区域内 - 限制感知能力
                # 无法获取自身绝对位置信息，使用默认值-1.0表示信息不可用
                obs[0] = -1.0  # x坐标不可用
                obs[1] = -1.0  # y坐标不可用
                # 速度信息仍可获取（不依赖外部定位系统）
                obs[2] = (m_uavs[i].m_v - 2) / 4.  # 速度信息仍可获取
                # 航向角信息仍可获取（可通过惯性导航获得）
                obs[3] = (m_uavs[i].m_w+PI/4) / (PI)  # 航向角信息仍可获取
                
                # 目标位置信息不可用（失去通信和导航信号）
                obs[44] = 0.0
                obs[45] = 0.0
                
                # 邻居信息不可用（失去通信能力）
                for j in range(8):
                    obs[4 + j * 4] = -1.0  # 邻居x坐标不可用
                    obs[5 + j * 4] = -1.0  # 邻居y坐标不可用
                    obs[6 + j * 4] = 0.0   # 邻居速度不可用
                    obs[7 + j * 4] = 0.0   # 邻居航向角不可用
                
                # 障碍物信息不可用（失去环境地图）
                for k in range(2):
                    obs[36+k] = -1.0
                    obs[40+k] = -1.0
            else:
                # 在拒止区域外 - 拥有完整感知能力
                # 归一化x坐标（0-1），将坐标范围从[0,500]映射到[0,1]
                obs[0] = (m_uavs[i].m_x) / 500.
                # 归一化y坐标（0-1），将坐标范围从[0,500]映射到[0,1]
                obs[1] = (m_uavs[i].m_y) / 500.
                # 归一化速度（0-1），将速度范围从[2,6]映射到[0,1]
                obs[2] = (m_uavs[i].m_v - 2) / 4.
                # 归一化航向角（0-1），将角度范围从[-π/4,3π/4]映射到[0,1]
                obs[3] = (m_uavs[i].m_w+PI/4) / (PI)
                
                # 初始化距离字典，用于存储与其他无人机的距离信息
                index_dist = {}
                # 计算与所有其他无人机的欧氏距离
                # 但排除已进入拒止区域的无人机（因为它们已失去通信能力）
                for j in range (self.m_uav_cnt):
                    # 跳过自身
                    if (j == i) : continue
                    # 跳过已坠毁的无人机
                    if(m_die[j]): continue
                    # 跳过已进入拒止区域的无人机（它们已失去通信能力）
                    if(uav_denied_status.get(j, False)): continue

                    if j not in m_uavs: continue
                    # 计算与无人机j的欧氏距离
                    dist = l2norm((m_uavs[j].m_x - m_uavs[i].m_x), (m_uavs[j].m_y - m_uavs[i].m_y ))
                    # 存储距离信息，以无人机ID为键，距离为值
                    index_dist[j] = dist
                # 按距离升序排序，最近的无人机排在前面
                index_dist = sorted(index_dist.items(),  key=lambda d: d[1], reverse=False)
                # 转换为字典格式
                index_dist = dict(index_dist)
                # 获取排序后的距离值列表
                value = sorted(index_dist.values())
                # 处理最近的8架无人机信息
                for j in range (8):
                    # 检查是否在通信范围内（UAV_UAV_COMM）
                    if j < len(index_dist) and value[j] < UAV_UAV_COMM:
                        # 获取距离第j近的无人机ID
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
                        # 超出通信范围或没有更多邻居，填充默认值
                        obs[4 + j * 4] = -1.0
                        obs[5 + j * 4] = -1.0
                        obs[6 + j * 4] = 0.0
                        obs[7 + j * 4] = 0.0
                # 归一化障碍物x坐标，将坐标范围从[0,500]映射到[0,1]
                for k in range (2):
                    obs[36+k] = (m_obstacless[k].x ) / 500
                # 归一化障碍物y坐标，将坐标范围从[0,500]映射到[0,1]
                for k in range (2):
                    obs[40+k] = (m_obstacless[k].y) / 500
                # 目标点相对位置x，将坐标范围从[400,500]映射到[-1,1]再映射到[0,2]
                obs[44] = (m_target.x - 450.) / 50.
                # 目标点相对位置y，将坐标范围从[400,500]映射到[-1,1]再映射到[0,2]
                obs[45] = (m_target.y - 450.) / 50.
                
            # 坠毁状态（0/1），表示当前无人机是否已坠毁
            obs[46] = int(m_die[i])
            # 拒止区域状态标记（0/1），表示当前无人机是否在拒止区域内
            # 这个信息对DRL算法的决策非常重要
            obs[47] = int(uav_denied_status.get(i, False))
            # 将当前无人机的观测数据添加到结果列表中
            result.append(obs)
            
        # 返回所有无人机的观测数据数组
        return np.array(result)

    # 添加属性访问器，以便外部代码可以访问uav_denied_status等属性
    @property
    def cpp_env(self):
        """提供对环境内部状态的访问"""
        # 直接返回self，这样外部代码可以通过self._env._env.uav_denied_status访问
        return self
    
    # 添加直接属性访问，避免AttributeError
    @property
    def uav_denied_status(self):
        """返回拒止区域状态字典"""
        return uav_denied_status
    
    @property
    def uav_inertial_positions(self):
        """返回惯性导航估计位置字典"""
        return uav_inertial_positions
    
    @property
    def uav_led_status(self):
        """返回LED状态字典"""
        return uav_led_status
    
    @property
    def uav_pointclouds(self):
        """返回点云数据字典"""
        return uav_pointclouds

    def getLidarPointcloudForUAV(self, uav_id):
        """
        为特定无人机获取激光雷达点云数据
        根据规范，在拒止区域内只使用点云数据进行处理，不需要生成RGB图像
        该方法直接调用激光雷达模拟器生成场景点云，然后过滤掉障碍物点云，只保留无人机点云
        
        参数:
            uav_id: 无人机ID
            
        返回:
            pointcloud: 点云数据 (N, 4) [x, y, z, reflectance]
                     只包含其他无人机的点云数据，不包含障碍物点云
        """
        # 只为未坠毁的无人机生成点云数据
        if m_die[uav_id]:  
            # 对于坠毁的无人机，提供空的点云数据
            return np.empty((0, 4))
            
        # 设置激光雷达姿态（基于无人机位置和朝向）
        # quaternion: 四元数表示旋转姿态，[0, 0, 0, 1]表示无旋转
        # translation: 三维位置坐标，z=1.5为飞行高度
        pose = {
            'quaternion': [0, 0, 0, 1],  # 简化为单位四元数
            'translation': [m_uavs[uav_id].m_x, m_uavs[uav_id].m_y, 1.5]  # 无人机位置，z=1.5为飞行高度
        }
        
        # 处理场景生成点云数据
        # 激光雷达通过主动发射激光并接收反射信号来探测周围环境
        # 因此即使在拒止区域内也不会失去探测能力
        if self.lidar_simulator is not None:
            # 收集环境中其他无人机的位置信息（用于生成点云）
            uav_positions = []
            for j in range(self.m_uav_cnt):
                # 排除自身和已坠毁的无人机，只保留其他友机的位置信息
                if j != uav_id and not m_die[j]:
                    uav_positions.append([m_uavs[j].m_x, m_uavs[j].m_y, 1.5])  # z=1.5为飞行高度
                
            # 始终基于真实位置生成点云数据，不生成模拟点云
            # 无论无人机是否在拒止区域内，都使用真实的无人机位置信息生成点云

            # 将其他无人机的位置信息传递给激光雷达模拟器
            self.lidar_simulator.uav_positions = uav_positions
            
            # 调用专门获取无人机点云的方法，获取环境中其他友机的点云数据
            # 通过本机模拟激光雷达扫描这些点云来识别其他无人机友机
            pointcloud = self.lidar_simulator.get_uav_pointcloud(pose)
            return pointcloud
        else:
            # 如果没有激光雷达模拟器，返回空的点云数据
            return np.empty((0, 4))

    # 计算奖励值
    def getRewards(self):
        # 声明全局碰撞标志
        global collisiono
        # 初始化奖励数组
        result = [0.0]*self.m_uav_cnt
        # 初始化本步坠毁状态
        die_on_this_step =[False]*self.m_uav_cnt
        # 遍历所有无人机
        for i in range (self.m_uav_cnt) :
            # 跳过已坠毁无人机
            if(m_die[i]):
                result[i] = 0
                continue
            # 重置路径长度
            self.sumway[i] = 0
            # 基础惩罚（时间步惩罚）
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
            for k in range (2):
                # 检查与障碍物距离
                if(l2norm(m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) < 30+k*5):
                    collisiono = True
                    break
            # 障碍物碰撞处理
            if(collisiono):
                # 碰撞惩罚（适度降低以减少波动）
                result[i] -= 50
                # 标记本步坠毁
                die_on_this_step[i] = True
            # 检查接近障碍物
            for k in range (2):
                # 检查危险距离
                if (k > 1):
                    # 检查危险距离
                    if (l2norm(m_next_pos[i][0] - m_obstacless[k].x,
                               m_next_pos[i][1] - m_obstacless[k].y) > 30 + k * 5 and l2norm(
                            m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) < 40 + k * 5):
                        # 接近惩罚
                        result[i] -= 2

            # 集群奖励 - 增强无人机聚集性
            # 计算与所有其他无人机的平均距离
            total_dist = 0.0
            alive_neighbors = 0

            for j in range(self.m_uav_cnt):
                # 跳过已坠毁
                if(m_die[j]): continue
                # 跳过自身
                if i == j: continue
                # 跳过在拒止区域内的无人机（它们采用不同的策略）
                if uav_denied_status.get(j, False): continue

                # 计算无人机间距离
                dist = l2norm(m_next_pos[j][0] - m_next_pos[i][0], m_next_pos[j][1] - m_next_pos[i][1])
                total_dist += dist
                alive_neighbors += 1

            # 额外的近距离奖励，鼓励无人机之间保持紧密联系
            for j in range(self.m_uav_cnt):
                # 跳过已坠毁
                if(m_die[j]): continue
                # 跳过自身
                if i == j: continue
                # 跳过在拒止区域内的无人机
                if uav_denied_status.get(j, False): continue
                
                # 计算无人机间距离
                dist = l2norm(m_next_pos[j][0] - m_next_pos[i][0], m_next_pos[j][1] - m_next_pos[i][1])
                # 在非常近的距离内给予额外奖励
                if 3 < dist < 15:
                    result[i] += 1.0

            # 到达目标区域
            if (l2norm(m_next_pos[i][0] - m_target.x, m_next_pos[i][1] - m_target.y) < 20) :
                # 到达奖励（适度降低以减少波动）
                result[i] += 150
                # 记录结束步数
                if (die_on_this_step[i] == False):
                    self.endstep[i] = self.m_steps
                # 标记本步坠毁
                die_on_this_step[i] = True
                
            # 添加信号质量衰减惩罚和路径持久性与相似性奖励
            # 更新无人机历史位置
            uav_id = i
            if uav_id not in uav_histories:
                uav_histories[uav_id] = []
            
            # 检查无人机是否在拒止区域内
            in_denied_area = False
            for denied_area_info in m_denied_areas:
                # 检查无人机是否在拒止区域内
                if is_in_denied_area((m_next_pos[i][0], m_next_pos[i][1]), denied_area_info['polygon']):
                    in_denied_area = True
                    break

            # 只在拒止区域外记录历史位置，并限制用于奖励计算的历史记录长度
            if not in_denied_area:
                uav_histories[uav_id].append((m_next_pos[i][0], m_next_pos[i][1]))
                # 限制用于奖励计算的历史记录长度，避免无限增长影响性能
                # 但不影响用于路径绘制的完整历史记录uav_positions_history
                if len(uav_histories[uav_id]) > HISTORY_LENGTH:
                    uav_histories[uav_id].pop(0)  # 移除最旧的位置记录

            total_penalty = 0.0
            for denied_area_info in m_denied_areas:
                penalty = signal_attenuation_penalty(
                    (m_next_pos[i][0], m_next_pos[i][1]), 
                    denied_area_info['polygon'], 
                    R_TRANS, 
                    K_ATT
                )
                # 添加平滑因子，避免惩罚值突变
                total_penalty += penalty  # 增强惩罚的影响因子
                # print(f"total_penalty:{total_penalty}")
            result[i] += total_penalty
            
            # 计算路径平滑奖励
            # 只有在拒止区域外才计算路径平滑奖励
            if uav_id in uav_histories and not uav_denied_status.get(i, False):
                smooth_reward = path_smoothness_reward(
                    uav_histories[uav_id],
                    K_SMOOTH
                )
                result[i] += smooth_reward
                
        # 更新坠毁状态
        for i in range (self.m_uav_cnt) :
            if die_on_this_step[i]:
                m_die[i] = True
                
        # 检查是否所有无人机都进入拒止区域，如果是则给予较大的惩罚
        all_in_denied_area = True  # 检查是否所有无人机都进入拒止区域
        
        # 检查所有无人机是否到达目标区域或仍在运行
        for i in range(self.m_uav_cnt):
            # 确保索引在有效范围内
            if i < len(die_on_this_step):
                # 如果有任何无人机未到达目标区域，则不结束
                if not die_on_this_step[i]:
                    all_done = False
            else:
                # 如果索引超出范围，默认认为未完成
                all_done = False
                break
                
        # 检查是否所有无人机都进入拒止区域
        for i in range(self.m_uav_cnt):
            if not uav_denied_status.get(i, False):
                all_in_denied_area = False
                break
        
        # 结束条件：所有无人机都到达目标区域、达到最大步数或所有无人机都进入拒止区域
        done = all_done or (self.m_steps % 200 == 0) or all_in_denied_area
        
        # 如果所有无人机都进入拒止区域，给予较大的惩罚
        if all_in_denied_area:
            # 给所有无人机施加较大的惩罚，鼓励避免全部进入拒止区域
            for i in range(self.m_uav_cnt):
                if not m_die[i]:  # 只对未坠毁的无人机施加惩罚
                    result[i] -= 100  # 给予较大的惩罚
                    
        # 返回奖励数组
        return result

    # 检查是否结束
    def isDone(self):
        all_done = True
        all_in_denied_area = True  # 检查是否所有无人机都进入拒止区域
        
        # 检查所有无人机是否到达目标区域或仍在运行
        for i in range(self.m_uav_cnt):
            # 确保索引在有效范围内
            if i < len(die_on_this_step):
                # 如果有任何无人机未到达目标区域，则不结束
                if not die_on_this_step[i]:
                    all_done = False
            else:
                # 如果索引超出范围，默认认为未完成
                all_done = False
                break
                
        # 检查是否所有无人机都进入拒止区域
        for i in range(self.m_uav_cnt):
            if not uav_denied_status.get(i, False):
                all_in_denied_area = False
                break
        
        # 结束条件：所有无人机都到达目标区域、达到最大步数或所有无人机都进入拒止区域
        done = all_done or (self.m_steps % 200 == 0) or all_in_denied_area
        
        # 如果所有无人机都进入拒止区域，给予较大的惩罚
        if all_in_denied_area:
            for i in range(self.m_uav_cnt):
                # 给所有无人机施加惩罚
                if not m_die[i]:  # 只对未坠毁的无人机施加惩罚
                    # 在奖励函数中添加较大的负奖励
                    # 这将在训练中鼓励无人机避免全部进入拒止区域
                    pass  # 实际惩罚在getRewards中实现
        
        return done

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

    # 获取无人机位置历史
    def getUavPositionsHistory(self):
        # 返回完整的无人机位置历史记录，用于路径绘制
        # 不限制历史记录长度，确保绘图功能正常工作
        return uav_positions_history

    def is_in_denied_area(self, x, y):
        """
        检查指定坐标是否在拒止区域内
        
        参数:
            x: x坐标
            y: y坐标
            
        返回:
            bool: True表示在拒止区域内，False表示在拒止区域外
        """
        # 遍历所有拒止区域
        for denied_area_info in m_denied_areas:
            # 使用is_in_denied_area函数检查是否在当前拒止区域内
            if is_in_denied_area((x, y), denied_area_info['polygon']):
                return True
        return False

# 信号质量衰减相关参数
R_TRANS = 70.0  # 信号衰减过渡区半径，增加过渡区域使惩罚更平滑
K_ATT = 5     # 惩罚权重系数，增强惩罚强度以更好地激励回避
K_PERSISTENCE = 5.0  # 路径持久性奖励系数
HISTORY_LENGTH = 15  # 增加路径历史长度以更好地计算平滑性
K_SMOOTH = 1.0  # 提高路径平滑奖励系数，更好地鼓励平滑路径

# 拒止区域参数
DENIED_AREA_RADIUS = 50  # 拒止区域默认半径

# 拒止区域类型枚举
DENIED_AREA_TYPE = {
    'CUSTOM': 0,         # 自定义拒止区域
    'OBSTACLE_BASED': 1  # 基于障碍物的拒止区域
}

def sat_function(x: float) -> float:
    """
    饱和函数，用于信号质量衰减计算
    
    该函数将输入值映射到[0,1]区间，用于计算信号质量因子
    工作原理类似于神经网络中的激活函数
    
    参数:
        x: 输入值，通常为无人机到拒止区域边界的距离与过渡区半径的比值
        
    返回:
        float: 饱和后的值，在[0,1]区间内
               - 当x<0时，返回0（无信号）
               - 当0<=x<=1时，返回x（信号质量随距离线性变化）
               - 当x>1时，返回1（信号质量良好）
    """
    # 当输入值小于0时，表示无人机深入拒止区域内部，信号质量为0
    if x < 0:
        return 0.0
    # 当输入值在[0,1]区间时，信号质量与距离成正比
    # 距离边界越近，信号质量越好
    elif 0 <= x <= 1:
        return x
    # 当输入值大于1时，表示无人机远离拒止区域，信号质量良好
    else:
        return 1.0


def is_in_denied_area(
    drone_pos: tuple,       # 无人机当前位置 (x, y)
    denied_area: Polygon    # 拒止区域多边形对象
) -> bool:
    """
    判断无人机是否在拒止区域内
    
    拒止区域是指由于信号干扰、敌方控制或其他原因导致无人机无法正常通信和导航的区域
    在拒止区域内，无人机将失去以下能力：
    1. 无法获取自身精确位置信息
    2. 无法获取障碍物精确位置信息
    3. 无法与其他无人机通信获取邻居信息
    4. 无法获取目标位置信息
    
    但无人机仍能通过以下方式感知环境：
    1. 激光雷达观测（主动传感器，不依赖外部信号）
    2. 速度和航向角信息（可通过惯性导航获得）
    
    参数:
        drone_pos: 无人机当前位置 (x, y) 坐标元组
        denied_area: 表示拒止区域边界的多边形对象
        
    返回:
        bool: True表示无人机在拒止区域内，False表示在拒止区域外
    """
    # 创建一个点对象表示无人机当前位置
    drone_point = Point(drone_pos)
    # 使用Shapely库的contains和touches方法判断点是否在多边形内或边界上
    # contains: 点在多边形内部
    # touches: 点在多边形边界上
    return denied_area.contains(drone_point) or denied_area.touches(drone_point)


def signal_attenuation_penalty(
    drone_pos: tuple,          # 无人机当前位置(x, y)
    denied_area: Polygon,      # 拒止区域几何边界
    r_trans: float,            # 信号衰减过渡区半径
    k_att: float               # 惩罚权重系数
) -> float:
    """
    信号质量衰减的连续惩罚函数
    该函数用于模拟无人机在接近拒止区域时受到的信号质量影响
    
    激励无人机主动规避进入拒止环境
    
    工作原理:
    1. 判断无人机是否在拒止区域内
    2. 只有在拒止区域外且距离边界小于r_trans时，才根据距离计算信号质量因子
    3. 根据信号质量因子计算惩罚值
    
    注意：根据系统设计，一旦无人机进入拒止区域，将完全失去位置信息
    因此，此函数只在无人机在拒止区域外且在影响范围内时根据距离计算惩罚
    当无人机在拒止区域内时，返回固定的惩罚值
    
    参数:
        drone_pos: 无人机当前位置 (x, y)
        denied_area: 拒止区域的多边形边界表示
        r_trans: 信号衰减过渡区半径，表示从完全信号到无信号的过渡距离
        k_att: 惩罚权重系数，控制惩罚的强度
        
    返回:
        float: 惩罚值，负数表示惩罚
    """
    # 创建无人机位置点对象
    drone_point = Point(drone_pos)
    
    # 判断无人机是否在拒止区域内
    in_denied_area = denied_area.contains(drone_point) or denied_area.touches(drone_point)
    
    # 只有在拒止区域外才计算基于距离的信号质量
    if not in_denied_area:
        # 计算无人机到拒止区域边界的欧几里得距离d_i
        # 使用Shapely库的distance方法计算点到多边形边界的最短距离
        d_i = drone_point.distance(denied_area.boundary)

        # 只有当距离小于过渡区半径r_trans时才触发惩罚机制
        # 这样可以避免无人机在距离拒止区域过远时也受到不必要的惩罚
        if d_i < r_trans:
            # 计算信号质量因子q_i^t
            # 使用sat_function函数将距离映射到[0,1]区间
            # q_i^t = sat(d_i / r_trans)
            # 当无人机远离拒止区域时，q_i^t接近1（信号质量好）
            # 当无人机接近拒止区域边界时，q_i^t逐渐减小（信号质量变差）
            q_i_t = sat_function(d_i / r_trans)

            # print(f"拒止区域外距离：{d_i}，惩罚值：{-k_att * (1 - q_i_t)}")
            # 计算连续惩罚项 R_att^t(i) = -k_att·(1 - q_i^t)
            # 惩罚值与信号质量因子成反比
            # 当信号质量好时(q_i^t接近1)，惩罚接近0
            # 当信号质量差时(q_i^t接近0)，惩罚接近-k_att
            return -k_att * (1 - q_i_t)
        else:
            # 当距离大于过渡区半径时，不施加惩罚
            return 0.0
    else:
        # 当无人机在拒止区域内时，返回更强的惩罚值
        # 模拟完全失去信号的情况，并激励尽快离开
        # 根据深入拒止区域的程度增加惩罚
        distance_into_denied = drone_point.distance(denied_area.centroid)
        penalty_multiplier = 2.0 + min(distance_into_denied / 10.0, 3.0)  # 惩罚倍数在2.0到5.0之间
        # print(f"拒止区域内，惩罚值：{-penalty_multiplier * k_att}")
        return -penalty_multiplier * k_att

def path_smoothness_reward(
    drone_history: list,
    k_smooth: float
) -> float:
    """
    改进的路径平滑奖励函数
    综合考虑转向角度、路径曲率和加速度变化，提供更稳定的路径平滑奖励
    
    工作原理:
    1. 计算转向角度平滑度（主要指标）
    2. 计算路径曲率平滑度（避免急转弯）
    3. 计算加速度变化平滑度（避免速度突变）
    4. 综合三个指标计算最终奖励
    
    参数:
        drone_history: 无人机位置历史 [(x, y), ...]
        k_smooth: 平滑奖励系数
        
    返回:
        float: 平滑奖励值，值越大表示路径越平滑
    """
    # 如果历史位置点数少于4个，则无法进行完整计算
    if len(drone_history) < 4:
        return 0.0
    
    # 使用最近10个点进行计算（如果历史记录足够长）
    history_length = min(len(drone_history), 10)
    start_idx = len(drone_history) - history_length
    
    # 1. 转向角度平滑度计算
    turn_angles = []
    for i in range(history_length - 2):
        idx = start_idx + i
        p1 = np.array(drone_history[idx])
        p2 = np.array(drone_history[idx+1])
        p3 = np.array(drone_history[idx+2])
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 1e-8 and norm_v2 > 1e-8:
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            turn_angles.append(np.degrees(angle))
    
    # 2. 路径曲率计算（使用连续4个点）
    curvatures = []
    for i in range(history_length - 3):
        idx = start_idx + i
        p1 = np.array(drone_history[idx])
        p2 = np.array(drone_history[idx+1])
        p3 = np.array(drone_history[idx+2])
        p4 = np.array(drone_history[idx+3])
        
        # 计算三个连续段的转向角度变化
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        if (np.linalg.norm(v1) > 1e-8 and np.linalg.norm(v2) > 1e-8 and 
            np.linalg.norm(v3) > 1e-8):
            # 计算相邻转向角度的变化
            cos1 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos2 = np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))
            
            angle1 = np.arccos(np.clip(cos1, -1.0, 1.0))
            angle2 = np.arccos(np.clip(cos2, -1.0, 1.0))
            
            # 曲率 = 转向角度变化率
            curvature = abs(angle2 - angle1)
            curvatures.append(np.degrees(curvature))
    
    # 3. 加速度变化计算
    accelerations = []
    for i in range(history_length - 2):
        idx = start_idx + i
        p1 = np.array(drone_history[idx])
        p2 = np.array(drone_history[idx+1])
        p3 = np.array(drone_history[idx+2])
        
        # 计算速度向量
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算加速度（速度变化率）
        if np.linalg.norm(v1) > 1e-8 and np.linalg.norm(v2) > 1e-8:
            acc = np.linalg.norm(v2) - np.linalg.norm(v1)
            accelerations.append(abs(acc))
    
    # 计算各项指标的平滑度得分
    turn_smoothness = 0.0
    curvature_smoothness = 0.0
    acc_smoothness = 0.0
    
    if turn_angles:
        avg_turn = np.mean(turn_angles)
        # 转向角度越小越平滑，使用指数衰减
        turn_smoothness = np.exp(-avg_turn / 30.0)  # 30度作为参考基准
    
    if curvatures:
        avg_curvature = np.mean(curvatures)
        # 曲率变化越小越平滑
        curvature_smoothness = np.exp(-avg_curvature / 15.0)  # 15度变化作为基准
    
    if accelerations:
        avg_acc = np.mean(accelerations)
        # 加速度变化越小越平滑
        acc_smoothness = np.exp(-avg_acc / 2.0)  # 2单位加速度变化作为基准
    
    # 综合平滑度得分（加权平均）
    if turn_angles and curvatures and accelerations:
        smoothness_score = 0.5 * turn_smoothness + 0.3 * curvature_smoothness + 0.2 * acc_smoothness
    elif turn_angles and curvatures:
        smoothness_score = 0.6 * turn_smoothness + 0.4 * curvature_smoothness
    elif turn_angles:
        smoothness_score = turn_smoothness
    else:
        smoothness_score = 0.0
    
    # 最终奖励值
    smooth_reward = k_smooth * smoothness_score
    return smooth_reward

def cluster_cohesion_reward(
    drone_positions: dict,
    uav_id: int
) -> float:
    """
    计算集群聚集性奖励
    
    参数:
        drone_positions: 所有无人机位置字典 {uav_id: (x, y)}
        uav_id: 当前无人机ID
        
    返回:
        float: 聚集性奖励值
    """
    # 获取当前无人机位置
    own_position = drone_positions.get(uav_id)
    if own_position is None:
        return 0.0
    
    # 获取其他无人机位置
    other_positions = {uid: pos for uid, pos in drone_positions.items() if uid != uav_id}
    
    if not other_positions:
        return 0.0  # 没有其他无人机
    
    # 计算到其他无人机的距离
    distances = []
    for other_pos in other_positions.values():
        dist = l2norm(own_position[0] - other_pos[0], own_position[1] - other_pos[1])
        distances.append(dist)
    
    # 计算平均距离
    avg_distance = np.mean(distances) if distances else 0.0
    
    # 理想聚集距离：20-40单位
    if 20 <= avg_distance <= 40:
        # 理想距离范围，给予最大奖励
        return 1.5
    elif avg_distance < 20:
        # 距离过近，给予惩罚
        return max(0, 1.5 - (20 - avg_distance) * 0.1)
    else:
        # 距离过远，给予惩罚
        return max(0, 1.5 - (avg_distance - 40) * 0.05)

def formation_keeping_reward(
    drone_positions: dict,
    uav_id: int
) -> float:
    """
    计算编队保持奖励
    
    参数:
        drone_positions: 所有无人机位置字典 {uav_id: (x, y)}
        uav_id: 当前无人机ID
        
    返回:
        float: 编队保持奖励值
    """
    # 获取当前无人机位置
    own_position = drone_positions.get(uav_id)
    if own_position is None:
        return 0.0
    
    # 获取其他无人机位置
    other_positions = {uid: pos for uid, pos in drone_positions.items() if uid != uav_id}
    
    if len(other_positions) < 2:
        return 0.0  # 需要至少2个其他无人机才能形成编队
    
    # 计算群体中心
    center_x = sum(pos[0] for pos in other_positions.values()) / len(other_positions)
    center_y = sum(pos[1] for pos in other_positions.values()) / len(other_positions)
    
    # 计算到群体中心的距离
    dist_to_center = l2norm(own_position[0] - center_x, own_position[1] - center_y)
    
    # 计算群体半径（最大距离）
    max_radius = 0.0
    for pos in other_positions.values():
        radius = l2norm(pos[0] - center_x, pos[1] - center_y)
        if radius > max_radius:
            max_radius = radius
    
    # 理想编队位置：在群体半径的60-80%范围内
    ideal_min_radius = max_radius * 0.6
    ideal_max_radius = max_radius * 0.8
    
    if ideal_min_radius <= dist_to_center <= ideal_max_radius:
        # 理想位置，给予最大奖励
        return 1.0
    elif dist_to_center < ideal_min_radius:
        # 过于靠近中心，给予惩罚
        return max(0, 1.0 - (ideal_min_radius - dist_to_center) * 0.2)
    else:
        # 过于远离中心，给予惩罚
        return max(0, 1.0 - (dist_to_center - ideal_max_radius) * 0.1)
