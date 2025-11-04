import numpy as np
from sklearn.cluster import KMeans
import torch as T
import math
import os

# 设置OpenMP线程数以避免Intel libiomp和LLVM libomp库冲突
# 当同时加载这两个不兼容的OpenMP库时，可能导致程序随机崩溃或死锁
# 将OMP_NUM_THREADS设置为1可以缓解此问题
# 添加更多环境变量设置以进一步减少OpenMP库冲突
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

class ClusterFollower:
    """
    聚类跟随模块
    在拒止区域内，无人机通过雷达点云数据进行聚类分析，识别其他无人机的位置，
    然后根据路径持久性和路径相似性判断最合适的跟随目标并跟随。
    """
    
    def __init__(self, max_uavs=8):
        """
        初始化聚类跟随模块
        :param max_uavs: 最大无人机数量
        """
        self.max_uavs = max_uavs
        self.history_length = 10
        # 存储每架无人机的历史位置
        self.uav_histories = {}
        # 存储聚类结果
        self.cluster_results = {}
        
    def extract_uav_positions_from_lidar(self, lidar_data):
        """
        从雷达数据中提取无人机位置
        :param lidar_data: 雷达数据 (3888维，即36*36*3)
        :return: 无人机位置列表
        """
        if lidar_data is None or len(lidar_data) == 0:
            return []
            
        # 将雷达数据重塑为图像格式 (36, 36, 3)
        lidar_image = lidar_data.reshape(36, 36, 3)
        
        # 提取高反射率点（无人机反射率通常在0.5以上）
        high_reflectance_points = []
        for i in range(36):
            for j in range(36):
                # 检查RGB值，如果亮度较高则可能是无人机
                brightness = np.mean(lidar_image[i, j, :])
                # 阈值0.5：区分无人机（高反射率）和环境障碍物（低反射率）
                # 无人机反射率范围通常在0.8-1.0之间，障碍物反射率范围通常在0.1-0.3之间
                if brightness > 0.5:  
                    high_reflectance_points.append([i, j, brightness])
        
        # 如果高反射率点不足3个，则无法进行有效聚类
        if len(high_reflectance_points) < 3:  
            return []
            
        # 使用K-means聚类识别无人机位置
        points = np.array(high_reflectance_points)
        n_clusters = min(self.max_uavs, len(points))
        
        try:
            # 使用K-means算法对无人机点进行聚类
            # n_init=10: 运行10次不同的初始化，选择最佳结果
            # random_state=42: 固定随机种子，确保结果可重现
            # 解决OpenMP库冲突和Windows上的内存泄漏问题
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(points[:, :2])  # 只使用x,y坐标进行聚类
            
            # 获取聚类中心
            centers = kmeans.cluster_centers_
            
            # 将聚类中心转换为实际坐标（需要根据雷达参数进行转换）
            uav_positions = []
            for center in centers:
                # 简化处理：将图像坐标转换为相对坐标
                # 实际应用中需要根据雷达的具体参数进行坐标转换
                x_rel = (center[1] - 18) / 18.0  # 转换为[-1, 1]范围
                y_rel = (center[0] - 18) / 18.0  # 转换为[-1, 1]范围
                uav_positions.append([x_rel, y_rel])
                
            return uav_positions
        except Exception as e:
            print(f"聚类过程中出现错误: {e}")
            return []
    
    def update_uav_history(self, uav_id, position):
        """
        更新无人机历史位置
        :param uav_id: 无人机ID
        :param position: 当前位置 [x, y]
        """
        if uav_id not in self.uav_histories:
            self.uav_histories[uav_id] = []
            
        self.uav_histories[uav_id].append(position)
        
        # 保持历史记录长度
        if len(self.uav_histories[uav_id]) > self.history_length:
            self.uav_histories[uav_id].pop(0)
    
    def calculate_path_similarity(self, path, own_heading):
        """
        计算路径相似性（改进版本：考虑多时间尺度和方向一致性）
        :param path: 路径
        :param own_heading: 自身航向
        :return: 相似性得分（0-1）
        """
        if len(path) < 2:
            return 0.5  # 默认值
        
        # 改进：考虑不同时间尺度的运动模式
        short_term_similarity = self._calculate_short_term_similarity(path, own_heading)
        long_term_similarity = self._calculate_long_term_similarity(path, own_heading)
        
        # 加权综合：短期相似性更重要（60%），长期相似性作为参考（40%）
        similarity = 0.6 * short_term_similarity + 0.4 * long_term_similarity
        
        return similarity
    
    def _calculate_short_term_similarity(self, path, own_heading):
        """计算短期（最近3个点）运动相似性"""
        if len(path) < 2:
            return 0.5
        
        # 取最近3个位置点
        recent_positions = path[-min(3, len(path)):]
        
        # 计算短期运动向量
        if len(recent_positions) > 1:
            dx = recent_positions[-1][0] - recent_positions[-2][0]
            dy = recent_positions[-1][1] - recent_positions[-2][1]
            
            # 计算运动方向角度
            if abs(dx) < 0.1 and abs(dy) < 0.1:
                return 0.5  # 几乎静止
            
            target_angle = math.atan2(dy, dx)
            
            # 计算角度差异（考虑周期性）
            angle_diff = abs(target_angle - own_heading)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            
            # 角度差异越小，相似性越高
            similarity = 1.0 - (angle_diff / math.pi)
            return max(0, min(similarity, 1))
        
        return 0.5
    
    def _calculate_long_term_similarity(self, path, own_heading):
        """计算长期（最近10个点）运动趋势相似性"""
        if len(path) < 3:
            return 0.5
        
        # 取最近10个位置点
        recent_positions = path[-min(10, len(path)):]
        
        # 使用线性回归计算长期运动趋势
        if len(recent_positions) >= 3:
            x_coords = [pos[0] for pos in recent_positions]
            y_coords = [pos[1] for pos in recent_positions]
            
            # 计算线性回归斜率
            if len(set(x_coords)) > 1:  # 确保x坐标有变化
                slope, _ = np.polyfit(range(len(x_coords)), x_coords, 1)
                # 简化处理：使用x方向斜率代表整体趋势
                trend_angle = math.atan(slope)
                
                # 计算角度差异
                angle_diff = abs(trend_angle - own_heading)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                
                similarity = 1.0 - (angle_diff / math.pi)
                return max(0, min(similarity, 1))
        
        return 0.5
    
    def calculate_path_persistence(self, path):
        """
        计算路径持久性（改进版本：考虑多维度稳定性）
        :param path: 路径
        :return: 持久性得分（0-1）
        """
        if len(path) < 3:
            return 0.5  # 默认值
        
        # 计算多个维度的稳定性
        speed_stability = self._calculate_speed_stability(path)
        direction_stability = self._calculate_direction_stability(path)
        acceleration_stability = self._calculate_acceleration_stability(path)
        
        # 综合持久性得分
        persistence = 0.4 * speed_stability + 0.4 * direction_stability + 0.2 * acceleration_stability
        
        return persistence
    
    def _calculate_speed_stability(self, path):
        """计算速度稳定性"""
        if len(path) < 3:
            return 0.5
        
        # 计算速度序列
        speeds = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            speed = math.sqrt(dx**2 + dy**2)
            speeds.append(speed)
        
        # 计算速度的变异系数（标准差/均值）
        if len(speeds) > 1 and np.mean(speeds) > 0.1:
            cv = np.std(speeds) / np.mean(speeds)
            stability = math.exp(-cv)  # 变异系数越小越稳定
            return min(max(stability, 0), 1)
        
        return 0.5
    
    def _calculate_direction_stability(self, path):
        """计算方向稳定性"""
        if len(path) < 3:
            return 0.5
        
        # 计算方向角度序列
        angles = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:  # 避免除零
                angle = math.atan2(dy, dx)
                angles.append(angle)
        
        # 计算角度变化的稳定性
        if len(angles) > 1:
            # 计算相邻角度变化的方差
            angle_changes = []
            for i in range(1, len(angles)):
                diff = abs(angles[i] - angles[i-1])
                diff = min(diff, 2 * math.pi - diff)  # 处理周期性
                angle_changes.append(diff)
            
            if angle_changes:
                variance = np.var(angle_changes)
                stability = math.exp(-variance / 0.5)  # 调整衰减系数
                return min(max(stability, 0), 1)
        
        return 0.5
    
    def _calculate_acceleration_stability(self, path):
        """计算加速度稳定性"""
        if len(path) < 4:
            return 0.5
        
        # 计算加速度序列
        accelerations = []
        for i in range(2, len(path)):
            # 计算速度变化
            v1_dx = path[i][0] - path[i-1][0]
            v1_dy = path[i][1] - path[i-1][1]
            v2_dx = path[i-1][0] - path[i-2][0]
            v2_dy = path[i-1][1] - path[i-2][1]
            
            # 计算加速度
            acc_dx = v1_dx - v2_dx
            acc_dy = v1_dy - v2_dy
            acceleration = math.sqrt(acc_dx**2 + acc_dy**2)
            accelerations.append(acceleration)
        
        # 计算加速度的稳定性
        if len(accelerations) > 1:
            variance = np.var(accelerations)
            stability = math.exp(-variance / 2.0)  # 调整衰减系数
            return min(max(stability, 0), 1)
        
        return 0.5
    
    def find_best_follow_target(self, own_id, uav_positions):
        """
        简化的跟随目标选择策略：直接使用GWO算法的最优结果
        GWO已经通过综合适应度函数（距离、移动、聚集、编队、平滑性）找到了最优跟随位置
        无需重复计算路径持久性和相似性，避免冗余计算
        :param own_id: 自身ID
        :param uav_positions: 检测到的其他无人机位置
        :return: 最佳跟随目标位置
        """
        if not uav_positions:
            return None
            
        # 简化的目标选择：直接返回第一个可用的目标位置
        # GWO算法会在后续步骤中基于这些目标位置进行优化
        return uav_positions[0]
    
    def calculate_path_stability(self, path):
        """
        计算路径稳定性（位置变化的平滑度）
        :param path: 路径
        :return: 稳定性得分（0-1）
        """
        if len(path) < 2:
            return 0.5  # 默认值
        
        # 计算连续位置变化的平滑度
        position_changes = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            position_changes.append(math.sqrt(dx**2 + dy**2))
        
        # 计算变化量的方差，方差越小越稳定
        if len(position_changes) > 1:
            variance = np.var(position_changes)
            # 使用指数衰减函数计算稳定性得分
            stability = math.exp(-variance / 10.0)  # 调整衰减系数
            return min(max(stability, 0), 1)  # 限制在0-1范围内
        
        return 0.5
    
    def calculate_distance_factor(self, target_position, current_position):
        """
        计算距离因素，避免选择过远的目标
        :param target_position: 目标位置
        :param current_position: 当前位置
        :return: 距离因素（0-1）
        """
        dx = target_position[0] - current_position[0]
        dy = target_position[1] - current_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # 理想跟随距离范围：15-50单位
        if distance <= 50:
            # 在合理范围内，距离越近得分越高
            return 1.0 - (distance / 100.0)
        else:
            # 距离过远，得分降低
            return max(0, 1.0 - (distance - 50) / 100.0)
    
    def calculate_follow_action(self, own_position, target_position, own_velocity, own_heading):
        """
        计算跟随动作
        :param own_position: 自身位置 [x, y]
        :param target_position: 目标位置 [x, y]
        :param own_velocity: 自身速度
        :param own_heading: 自身航向角
        :return: 动作 [角度调整, 速度调整]
        """
        if target_position is None:
            return [0.0, 0.0]
            
        # 计算到目标的方向
        dx = target_position[0] - own_position[0]
        dy = target_position[1] - own_position[1]
        
        target_angle = math.atan2(dy, dx)
        
        # 计算角度差
        angle_diff = target_angle - own_heading
        # 规范化角度差到[-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # 计算距离
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 计算角度调整（归一化到[-1, 1]）
        angle_action = angle_diff / math.pi
        
        # 计算速度调整
        target_speed = min(6.0, max(2.0, distance * 2.0))  # 根据距离调整速度
        speed_action = (target_speed - own_velocity) / 4.0  # 归一化速度差
        speed_action = max(-1.0, min(1.0, speed_action))
        
        return [angle_action, speed_action]
    
    def process_denied_area_uav(self, uav_id, lidar_data, own_position, own_velocity, own_heading):
        """
        处理拒止区域内的无人机
        :param uav_id: 无人机ID
        :param lidar_data: 雷达数据
        :param own_position: 自身位置（可能不可用）
        :param own_velocity: 自身速度
        :param own_heading: 自身航向角
        :return: 动作 [角度调整, 速度调整]
        """
        # 从雷达数据中提取无人机位置
        uav_positions = self.extract_uav_positions_from_lidar(lidar_data)
        
        # 找到最佳跟随目标
        target_position = self.find_best_follow_target(uav_id, uav_positions)
        
        # 更新自身历史位置（使用相对位置）
        if own_position is not None:
            self.update_uav_history(uav_id, own_position)
        
        # 计算跟随动作
        action = self.calculate_follow_action(own_position, target_position, own_velocity, own_heading)
        
        return action