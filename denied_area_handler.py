import numpy as np
from sklearn.cluster import KMeans
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

class DeniedAreaHandler:
    """
    拒止区域处理器
    负责处理无人机进入拒止区域后的特殊行为
    """
    
    def __init__(self, max_uavs=8):
        """
        初始化拒止区域处理器
        :param max_uavs: 最大无人机数量
        """
        self.max_uavs = max_uavs
        self.history_length = 10
        # 存储每架无人机的历史位置
        self.uav_histories = {}
        # 存储聚类结果
        self.cluster_results = {}
        # 存储每架无人机当前跟随的目标
        self.current_follow_targets = {}
        
    def extract_uav_positions_from_pointcloud(self, pointcloud):
        """
        从点云数据中提取无人机位置
        参考点云数据结构：(N, 4) [x, y, z, reflectance]
        参考lidar01中的实现方法，直接处理点云数据而不是生成图像
        :param pointcloud: 点云数据 (N, 4) [x, y, z, reflectance]
        :return: 无人机位置列表 [(x, y), ...]
        """
        # 检查点云数据是否为空
        if pointcloud is None or len(pointcloud) == 0:
            return []
            
        # 提取高反射率点（无人机反射率通常在0.5以上）
        # 阈值0.5： 区分无人机（高反射率）和环境障碍物（低反射率）
        # 无人机反射率范围通常在0.8-1.0之间，障碍物反射率范围通常在0.1-0.3之间
        # 使用0.5作为分界值，将反射率高于0.5的点识别为无人机点，低于0.5的点识别为障碍物点
        drone_mask = pointcloud[:, 3] >= 0.5  # 使用标准阈值0.5来识别无人机，排除低反射率噪声
        drone_points = pointcloud[drone_mask]
        
        # 如果无人机点不足3个，则无法进行有效聚类
        # 降低要求以适应稀疏场景，最少只需要3个点即可进行聚类
        if len(drone_points) < 3:  
            return []
            
        # 使用K-means聚类识别无人机位置
        # 根据实际检测到的无人机点数量智能确定聚类数
        # 限制聚类数量不超过最大无人机数量，同时考虑每个聚类至少需要一定数量的点
        max_possible_clusters = len(drone_points) // 5  # 假设每个无人机至少有5个点
        n_clusters = min(self.max_uavs, max_possible_clusters, len(drone_points))
        
        # 确保至少有1个聚类
        n_clusters = max(1, n_clusters)
        
        try:
            # 使用K-means算法对无人机点进行聚类
            # n_init=10: 运行10次不同的初始化，选择最佳结果
            # random_state=42: 固定随机种子，确保结果可重现
            # 解决OpenMP库冲突和Windows上的内存泄漏问题
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
            # 当聚类失败时，尝试使用更简单的方法：选择反射率最高的点作为无人机位置
            # 这种方法虽然不如聚类精确，但可以避免程序崩溃
            try:
                # 根据实际可能的无人机数量选择反射率最高的几个点作为无人机位置
                max_possible_clusters = len(drone_points) // 5  # 假设每个无人机至少有5个点
                n_clusters = min(self.max_uavs, max_possible_clusters)
                n_clusters = max(1, n_clusters)  # 确保至少有1个聚类
                
                # 选择反射率最高的几个点作为无人机位置
                top_points = drone_points[drone_points[:, 3].argsort()[-min(n_clusters, len(drone_points)):]]
                uav_positions = [(point[0], point[1]) for point in top_points]
                return uav_positions
            except Exception as e2:
                print(f"备用方法也失败了: {e2}")
                return []
    
    def update_uav_history(self, uav_id, position):
        """
        更新无人机历史位置
        :param uav_id: 无人机ID
        :param position: 当前位置 (x, y)
        """
        if uav_id not in self.uav_histories:
            self.uav_histories[uav_id] = []
            
        self.uav_histories[uav_id].append(position)
        
        # 保持历史记录长度，移除最旧的位置记录
        if len(self.uav_histories[uav_id]) > self.history_length:
            self.uav_histories[uav_id].pop(0)
            
        # 添加平滑处理，避免位置突变
        if len(self.uav_histories[uav_id]) > 2:
            # 计算最近两次位置的变化
            prev_pos = self.uav_histories[uav_id][-2]
            curr_pos = self.uav_histories[uav_id][-1]
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            # 如果变化过大，可能是噪声，进行平滑处理
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 10:  # 如果距离变化超过10个单位，认为可能是噪声
                # 对当前位置进行平滑处理
                smoothed_x = prev_pos[0] + dx * 0.7
                smoothed_y = prev_pos[1] + dy * 0.7
                self.uav_histories[uav_id][-1] = (smoothed_x, smoothed_y)
    
    def calculate_path_similarity(self, path1, path2):
        """
        计算两条路径的相似性
        通过计算运动向量的余弦相似度来衡量路径方向的一致性
        :param path1: 路径1
        :param path2: 路径2
        :return: 相似性得分 [0, 1]，值越大表示越相似
        """
        # 如果路径点数不足2个，则无法计算运动向量
        if len(path1) < 2 or len(path2) < 2:
            return 0.0
            
        # 计算运动向量（位置差分）
        # np.diff函数计算相邻点之间的差值，得到运动向量
        vec1 = np.diff(np.array(path1), axis=0)
        vec2 = np.diff(np.array(path2), axis=0)
        
        # 对齐两个向量序列的长度，取较短的长度
        min_len = min(len(vec1), len(vec2))
        vec1, vec2 = vec1[:min_len], vec2[:min_len]
        
        # 计算余弦相似度
        cos_sim = []
        # 遍历每一对运动向量
        for v1, v2 in zip(vec1, vec2):
            # 计算向量的模长
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            # 避免除零错误
            if norm_v1 > 1e-8 and norm_v2 > 1e-8:
                # 计算余弦相似度：cos(θ) = (A·B)/(|A|*|B|)
                cos_sim.append(np.dot(v1, v2) / (norm_v1 * norm_v2))
            else:
                # 如果其中一个向量为零向量，则相似度为0
                cos_sim.append(0.0)
                
        # 返回平均余弦相似度
        return np.mean(cos_sim) if cos_sim else 0.0
    
    def calculate_path_persistence(self, path):
        """
        计算路径持久性（稳定性）
        通过计算运动向量的方差来衡量路径的稳定性，方差越小表示路径越稳定
        :param path: 路径
        :return: 持久性得分 [0, 1]，值越大表示越稳定
        """
        # 如果路径点数不足2个，则无法计算运动向量
        if len(path) < 2:
            return 0.0
            
        # 计算运动向量
        vectors = np.diff(np.array(path), axis=0)
        
        # 计算方差，方差越小表示越稳定
        # axis=0: 按列计算方差（分别计算x和y方向的方差）
        # sum(): 将x和y方向的方差相加
        var = np.var(vectors, axis=0).sum()
        
        # 持久性与方差成反比
        # 使用1/(1+var)将方差映射到[0,1]区间
        # 当方差为0时，持久性为1；当方差增大时，持久性趋近于0
        return 1.0 / (1.0 + var)
    
    def find_best_follow_target_by_led_frequency(self, pointcloud, own_position, uav_id):
        """
        基于LED闪烁频率（反射率）找到最佳跟随目标
        在拒止区域内，无人机通过识别拒止区域外无人机的LED闪烁频率来选择跟随目标
        LED关闭（低反射率）表示在拒止区域内，LED开启（高反射率）表示在拒止区域外
        
        参数:
            pointcloud: 点云数据 (N, 4) [x, y, z, reflectance]
            own_position: 自身位置 (x, y)
            uav_id: 无人机ID，用于跟踪当前跟随目标
            
        返回:
            target_position: 最佳跟随目标位置 (x, y)
        """
        # 检查点云数据是否为空
        if pointcloud is None or len(pointcloud) == 0:
            # 如果没有点云数据，清除当前跟随目标
            if uav_id in self.current_follow_targets:
                del self.current_follow_targets[uav_id]
            return None
            
        # 分离出高反射率点（表示拒止区域外的无人机）
        # 反射率阈值0.5用于区分LED开启和关闭状态
        # 高于0.5表示LED开启（拒止区域外），低于0.5表示LED关闭（拒止区域内）
        outside_drone_mask = pointcloud[:, 3] >= 0.5
        outside_drone_points = pointcloud[outside_drone_mask]
        
        # 根据反射率值识别无人机编号（每个无人机有唯一的反射率特征）
        # 反射率值在0.8-1.0范围内，每个无人机编号对应唯一的反射率值
        # 通过反射率值可以识别具体的无人机编号
        drone_id_by_reflectance = {}
        for point in outside_drone_points:
            x, y, z, reflectance = point
            # 根据反射率值估算无人机编号
            # 反射率 = 0.8 + (drone_id * 0.025)
            estimated_id = int((reflectance - 0.8) / 0.025)
            if 0 <= estimated_id < self.max_uavs:
                if estimated_id not in drone_id_by_reflectance:
                    drone_id_by_reflectance[estimated_id] = []
                drone_id_by_reflectance[estimated_id].append((x, y, z))
        
        # 如果没有拒止区域外的无人机（LED开启的无人机），则无法选择跟随目标
        if len(outside_drone_points) == 0:
            # 清除当前跟随目标
            if uav_id in self.current_follow_targets:
                del self.current_follow_targets[uav_id]
            return None
            
        # 检查是否已经有当前跟随目标且该目标仍然可见
        if uav_id in self.current_follow_targets:
            current_target = self.current_follow_targets[uav_id]
            # 检查当前目标是否仍在点云数据中
            target_found = False
            for i in range(len(outside_drone_points)):
                x, y = outside_drone_points[i, 0], outside_drone_points[i, 1]
                # 检查点是否接近当前跟踪的目标位置（考虑一定的误差范围）
                distance_to_target = math.sqrt((x - current_target[0])**2 + (y - current_target[1])**2)
                if distance_to_target < 5.0:  # 假设5米范围内认为是同一目标
                    target_found = True
                    # 更新目标位置（使用更精确的位置）
                    self.current_follow_targets[uav_id] = (x, y)
                    return (x, y)
            
            # 如果当前目标不再可见，清除它
            if not target_found:
                del self.current_follow_targets[uav_id]
        
        # 使用基于反射率识别的无人机编号进行精确跟随
        # 如果已经通过反射率识别到具体的无人机编号，优先使用这些信息
        if drone_id_by_reflectance:
            # 选择距离最近的可识别无人机作为跟随目标
            min_distance = float('inf')
            best_target = None
            best_drone_id = None
            
            for drone_id, points in drone_id_by_reflectance.items():
                # 计算该无人机所有点的中心位置
                center_x = np.mean([p[0] for p in points])
                center_y = np.mean([p[1] for p in points])
                
                # 计算到该无人机中心的距离
                distance = math.sqrt(center_x**2 + center_y**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_target = (center_x, center_y)
                    best_drone_id = drone_id
            
            if best_target is not None:
                # 记录跟随的无人机编号，用于后续跟踪
                self.current_follow_targets[uav_id] = best_target
                print(f"无人机 {uav_id} 通过LED闪烁频率识别到无人机 {best_drone_id}，正在跟随")
                return best_target
        
        # 如果无法通过反射率识别无人机编号，使用传统的聚类方法
        try:
            # 根据点的数量动态确定聚类数
            min_points_per_drone = 10  # 每个无人机至少需要的点数
            max_possible_clusters = len(outside_drone_points) // min_points_per_drone
            n_clusters = min(max(1, max_possible_clusters), self.max_uavs)
            
            if len(outside_drone_points) < min_points_per_drone:
                n_clusters = 1
            
            # 使用K-means聚类找到无人机中心点
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(outside_drone_points[:, :2])
            
            cluster_centers = kmeans.cluster_centers_
            
        except Exception as e:
            print(f"聚类过程中出现错误: {e}")
            cluster_centers = [np.mean(outside_drone_points[:, :2], axis=0)]
        
        # 如果只有一个聚类中心，直接选择它作为跟随目标
        if len(cluster_centers) == 1:
            target = (cluster_centers[0][0], cluster_centers[0][1])
            self.current_follow_targets[uav_id] = target
            return target
            
        # 如果有多个聚类中心，选择距离最近的一个作为跟随目标
        if own_position is not None:
            # 计算到各个聚类中心的距离
            distances = []
            for i, center in enumerate(cluster_centers):
                x, y = center[0], center[1]
                # 修正坐标计算：x,y是相对于LiDAR传感器的局部坐标，需要正确计算距离
                # 由于LiDAR安装在无人机上，直接使用局部坐标计算到目标的距离
                distance = math.sqrt(x**2 + y**2)  # 使用欧几里得距离计算
                # 综合评分：优先选择近距离的目标
                score = 1.0 / (1.0 + distance/10.0)  # 距离归一化因子10.0
                distances.append((score, (x, y)))
                
            # 按综合评分排序，选择评分最高的目标
            distances.sort(reverse=True)  # 降序排列
            target = distances[0][1]  # 最佳目标位置
            self.current_follow_targets[uav_id] = target
            return target
        else:
            # 如果没有自身位置信息，选择距离最近的目标
            min_distance = float('inf')
            best_target = None
            for i, center in enumerate(cluster_centers):
                x, y = center[0], center[1]
                distance = math.sqrt(x**2 + y**2)
                if distance < min_distance:
                    min_distance = distance
                    best_target = (x, y)
            if best_target is not None:
                self.current_follow_targets[uav_id] = best_target
            return best_target
    
    def find_best_follow_target(self, own_id, own_history, uav_positions):
        """
        根据路径持久性和相似性找到最佳跟随目标
        综合考虑路径相似性和持久性，选择最合适的跟随目标
        :param own_id: 自身ID
        :param own_history: 自身历史位置
        :param uav_positions: 检测到的其他无人机位置
        :return: 最佳跟随目标位置 (x, y)
        """
        # 如果没有检测到其他无人机，则无法进行跟随决策
        if not uav_positions:
            return None
            
        # 如果自身历史数据不足，跟随最近的无人机
        if len(own_history) < 2:
            # 计算到各个无人机的距离，选择最近的
            min_distance = float('inf')
            best_target = None
            for pos in uav_positions:
                distance = math.sqrt((pos[0] - own_history[-1][0])**2 + (pos[1] - own_history[-1][1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_target = pos
            return best_target
                    
        # 添加一个简单的启发式：如果检测到的目标太多，选择距离自身较远的目标
        # 这有助于避免在密集区域产生震荡行为
        if len(uav_positions) > 3:
            # 计算到各个无人机的距离，选择距离适中的目标
            distances = []
            for pos in uav_positions:
                distance = math.sqrt((pos[0] - own_history[-1][0])**2 + (pos[1] - own_history[-1][1])**2)
                distances.append((distance, pos))
            # 按距离排序，选择中间距离的目标
            distances.sort()
            # 选择距离排名第二的目标（避免选择最近的可能不稳定的目标）
            best_target = distances[min(1, len(distances)-1)][1]
            return best_target
            
        best_target = None
        best_score = -1.0
        
        # 遍历所有检测到的无人机位置
        for pos in uav_positions:
            # 为每个候选目标创建临时历史（包括当前位置）
            # 简化处理，实际应该有更长的历史
            temp_history = [pos]  
            
            # 计算路径相似性
            similarity = self.calculate_path_similarity(own_history, temp_history)
            
            # 计算路径持久性
            persistence = self.calculate_path_persistence(temp_history)
            
            # 综合评分：0.7权重给相似性，0.3权重给持久性
            # 这种加权方式更重视路径方向的一致性
            score = 0.7 * similarity + 0.3 * persistence
            
            # 更新最佳目标
            if score > best_score:
                best_score = score
                best_target = pos
                
        return best_target
    
    def calculate_follow_action(self, own_position, target_position, own_velocity, own_heading):
        """
        计算跟随动作
        根据目标位置计算角度调整和速度调整
        :param own_position: 自身位置 (x, y) - 世界坐标
        :param target_position: 目标位置 (x, y) - LiDAR局部坐标（相对于无人机当前位置和朝向）
                              x+: 前方, y+: 左方
        :param own_velocity: 自身速度
        :param own_heading: 自身航向角
        :return: 动作 [角度调整, 速度调整]，范围均为[-1, 1]
        """
        # 如果没有目标位置，则保持当前状态
        if target_position is None:
            return [0.0, 0.0]

        # 计算到目标的方向向量
        # target_position是LiDAR局部坐标系中的坐标，x+表示前方，y+表示左方
        local_dx = target_position[0]  # 前方距离
        local_dy = target_position[1]  # 左方距离
        
        # 在LiDAR局部坐标系中计算到目标的角度（相对于无人机前方）
        # atan2(y, x)返回的是相对于正x轴的角度，这里x是前方，y是左方
        target_angle_local = math.atan2(local_dy, local_dx)
        
        # 计算角度差（在局部坐标系中，无人机当前朝向为0度）
        angle_diff = target_angle_local
        # 规范化角度差到[-π, π]范围
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # 计算到目标的距离
        distance = math.sqrt(local_dx**2 + local_dy**2)
        
        # 计算角度调整（归一化到[-1, 1]）
        # 将角度差从[-π, π]映射到[-1, 1]
        # 添加平滑因子0.7，使动作更加平滑
        angle_action = max(-1.0, min(1.0, angle_diff / math.pi)) * 0.7
        
        # 计算速度调整
        # 根据距离调整目标速度，距离越远速度越快
        # 最小速度2.0，最大速度6.0
        target_speed = min(6.0, max(2.0, distance * 0.03))  # 降低速度增益因子，使动作更平滑
        # 计算速度差并归一化到[-1, 1]
        # 速度范围[2, 6]映射到[-1, 1]
        speed_action = (target_speed - own_velocity) / 4.0  
        # 添加平滑因子0.7，避免动作过于剧烈
        speed_action = max(-1.0, min(1.0, speed_action)) * 0.7
        
        return [angle_action, speed_action]
    
    def process_denied_area_uav(self, uav_id, pointcloud, own_position, own_velocity, own_heading, is_training=True):
        """
        处理拒止区域内的无人机
        整合所有步骤：提取位置、更新历史、找到最佳目标、计算动作
        :param uav_id: 无人机ID
        :param pointcloud: 点云数据
        :param own_position: 自身位置 (x, y)
        :param own_velocity: 自身速度
        :param own_heading: 自身航向角
        :param is_training: 是否处于训练模式，默认为True
        :return: 动作 [角度调整, 速度调整]
        """
        # 基于LED闪烁频率找到最佳跟随目标
        # 在拒止区域内，无人机通过识别拒止区域外无人机的LED闪烁频率来选择跟随目标
        # 传递uav_id以保持目标跟踪
        target_position = self.find_best_follow_target_by_led_frequency(pointcloud, own_position, uav_id)
        
        # 更新自身历史位置
        if own_position is not None:
            self.update_uav_history(uav_id, own_position)
        
        # 仅在测试模式下打印相关信息
        if not is_training and target_position is not None:
            print(f"无人机 {uav_id} 在拒止区域内通过LED闪烁频率识别到目标位置 {target_position}，正在执行跟随动作")
        
        # 计算跟随动作
        action = self.calculate_follow_action(own_position, target_position, own_velocity, own_heading)
        
        # 如果没有有效的跟随目标，使用基于速度和航向的保守策略
        if target_position is None:
            # 保持当前航向基本不变（角度调整为0）并轻微减速，直到驶离拒止区域
            # 根据经验教训，角度调整应为0，速度调整应为-0.1
            angle_action = 0.0  # 保持当前航向基本不变
            speed_action = -0.1  # 轻微减速至-0.1，直到驶离拒止区域
            action = [angle_action, speed_action]
            # 仅在测试模式下打印动作信息
            if not is_training:
                print(f"无人机 {uav_id} 在拒止区域内未检测到有效的跟随目标，采用保守策略：角度调整={action[0]:.3f}, 速度调整={action[1]:.3f}")
        else:
            # 仅在测试模式下打印动作信息
            if not is_training and action != [0.0, 0.0]:
                print(f"无人机 {uav_id} 在拒止区域内执行基于LED频率的跟随动作: 角度调整={action[0]:.3f}, 速度调整={action[1]:.3f}")
        
        return action
