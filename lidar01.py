import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R  # 用于四元数到旋转矩阵的转换
import random
matplotlib.use('TkAgg')


class EnvironmentObstacleSimulator:
    """环境障碍物点云生成器"""
    
    def __init__(self, n_bins=36, d_max=90.0, h=3.0, num_drones=8, env_obstacles=None):
        """
        初始化模拟器
        修改：使用环境中的真实障碍物
        """
        self.lidar_encoder = LidarEncoder(n_bins, d_max, h, num_drones)  # 使用新的h值
        self.num_drones = num_drones  # 无人机数量
        self.drone_params = []  # 存储每个无人机的参数（新增）
        # 从环境文件中获取障碍物位置
        self.env_obstacles = env_obstacles
        # 存储无人机反射率特征
        self.uav_reflectance_features = {}
        # 控制是否生成模拟点云的标志
        self.use_simulated_pointcloud = False
        # 拒止区域检查器（用于判断无人机是否在拒止区域内）
        self.denied_area_checker = None


    def generate_drone_pointcloud(self, drone_id, drone_positions=None, pose=None):
        """
        生成无人机友机的点云数据
        特点：
        - 球形形状（模拟无人机）
        - 通过LED闪烁频率识别无人机编号
        
        参数:
            drone_id: 无人机的唯一标识符
            drone_positions: 环境中其他无人机的位置信息 [(x, y, z), ...]
            pose: LiDAR传感器的姿态信息
            
        返回:
            drone_points: 带反射率属性的无人机点云
        """
        # 初始化默认参数
        size = 1.0  # 无人机半径为1
        reflectance = 0.8 + (drone_id * 0.025)  # 每个无人机编号有唯一的反射率值
        center = np.array([0, 0, 0])  # 默认中心位置
        
        # 检查是否需要基于真实位置生成点云
        # 当有无人机进入拒止区域时，我们基于其他无人机的真实位置生成点云
        # drone_positions 包含环境中其他无人机的位置信息
        if drone_positions is not None and len(drone_positions) > 0:
            # 获取真实无人机位置（注意：这里应该从drone_positions列表中选择合适的无人机，
            # 而不是直接使用drone_id作为索引）
            # 为简化实现，我们使用列表中的第一个无人机位置
            drone_pos = drone_positions[0] if len(drone_positions) > 0 else None
            if drone_pos is not None:
                x, y, z = drone_pos[0], drone_pos[1], drone_pos[2]
                center = np.array([x, y, z])
                size = 1.0  # 无人机半径为1
                
                # 检查该无人机是否在拒止区域内
                in_denied_area = False
                if hasattr(self, 'denied_area_checker') and self.denied_area_checker:
                    # 如果有拒止区域检查器，则检查该无人机是否在拒止区域内
                    in_denied_area = self.denied_area_checker(drone_pos[0], drone_pos[1])
        
                # 根据是否在拒止区域内设置LED状态
                # 如果在拒止区域内，LED关闭（反射率低）；否则根据编号设置闪烁频率（不同反射率）
                if in_denied_area:
                    # 在拒止区域内，LED关闭，反射率低
                    reflectance = 0.0  # 反射率为0表示LED完全关闭
                else:
                    # 在拒止区域外，LED开启，通过不同反射率表示不同闪烁频率
                    # 为每架无人机分配唯一的反射率值（0.8-1.0范围）
                    if hasattr(self, 'uav_reflectance_features') and drone_id in self.uav_reflectance_features:
                        reflectance = self.uav_reflectance_features[drone_id]
                    else:
                        reflectance = 0.8 + (drone_id * 0.025)  # 每个无人机编号有唯一的反射率值
            # 注意：如果drone_positions不为空但drone_pos为空，不生成任何点云
        else:
            # 如果没有提供无人机位置信息，则不生成任何点云
            # 返回空的点云数据
            return np.empty((0, 4))

        # 生成球形点云（模拟无人机形状，半径为1）
        num_points = 30  # 点数量
        # 在球体内生成随机点
        u = np.random.uniform(0, 1, num_points)
        v = np.random.uniform(0, 1, num_points)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        r = np.random.uniform(0, size, num_points)  # 半径为1
        x = r * np.sin(phi) * np.cos(theta) + center[0]
        y = r * np.sin(phi) * np.sin(theta) + center[1]
        z = r * np.cos(phi) + center[2]
        
        # 添加反射率属性（第四列）
        intensities = np.full(num_points, reflectance)
        drone_points = np.column_stack((x, y, z, intensities))
        
        return drone_points

    def generate_scene_pointcloud(self, pose):
        """
        生成整个场景的点云数据（仅包含无人机，不包含障碍物）
        """
        lidar_pos = np.array(pose['translation'])
        all_points = []
        self.drone_pointclouds = []  # 存储无人机点云

        # 生成无人机点云
        # 获取环境中其他无人机的真实位置
        drone_positions = []
        if hasattr(self, 'uav_positions') and self.uav_positions is not None:
            drone_positions = self.uav_positions

        # 只有当有无人机位置信息时才生成点云
        # 如果drone_positions为空，则不生成任何点云数据
        if len(drone_positions) > 0:
            # 为每个无人机生成点云数据，确保不超过实际无人机数量和可用位置数量的最小值
            for i in range(min(self.num_drones, len(drone_positions))):
                # 传递当前无人机ID、位置信息和姿态信息给generate_drone_pointcloud方法
                # drone_positions包含了环境中其他无人机的位置信息
                # i是当前要生成点云的无人机索引
                drone_points = self.generate_drone_pointcloud(i, [drone_positions[i]], pose)
                all_points.append(drone_points)
                self.drone_pointclouds.append(drone_points)
        else:
            # 如果没有无人机位置信息，则清空无人机点云列表
            self.drone_pointclouds = []

        # 转换为LiDAR坐标系
        lidar_points = [points[:, :3] - lidar_pos for points in all_points]  # 只转换坐标
        # 保留反射率属性（第四列）
        intensities = [points[:, 3] for points in all_points]
        
        # 合并坐标和反射率
        merged_points = []
        for coords, intensity in zip(lidar_points, intensities):
            merged_points.append(np.column_stack((coords, intensity)))
            
        return np.vstack(merged_points) if merged_points else np.array([])
    
    def get_uav_pointcloud(self, pose):
        """
        获取当前环境中其他无人机的点云数据
        该方法只生成并返回无人机的点云数据，不包含障碍物点云
        用于拒止区域内的无人机感知和跟随
        
        参数:
            pose: LiDAR传感器的姿态信息
            
        返回:
            drone_pointcloud: 无人机点云数据 (N, 4) [x, y, z, reflectance]
        """
        # 生成整个场景的点云
        scene_pointcloud = self.generate_scene_pointcloud(pose)

        
        # 只返回无人机点云（反射率>=0.5的点）
        # 使用0.5作为分界值，将反射率高于0.5的点识别为无人机点，低于0.5的点识别为障碍物点
        if len(scene_pointcloud) > 0:
            drone_mask = scene_pointcloud[:, 3] >= 0.5
            result = scene_pointcloud[drone_mask]
            return result
        else:
            return np.empty((0, 4))


class LidarEncoder:
    """
    LiDAR 数据编码器
    功能：
    - 处理LiDAR点云数据
    - 生成障碍地图
    - 维护历史数据以显示时间序列信息
    """

    def __init__(self, n_bins=36, d_max=10.0, h=3.0, num_drones=8):
        """
        初始化LiDAR编码器

        参数:
            n_bins: 角度分箱数（36表示每10度一个分箱）
            d_max: 最大检测距离
            h: LiDAR的垂直检测范围（从中心上下各h米）
            num_drones: 无人机数量
        """
        self.n_bins = n_bins  # 角度分箱数
        self.d_max = d_max  # 最大检测距离
        self.h = h  # 垂直检测范围
        self.num_drones = num_drones  # 无人机数量
        # 修改：历史数据扩展为三通道（RGB）
        # 确保维度正确：(时间帧, 角度分箱, 通道数) 以匹配可视化需求
        self.history = np.ones((36, n_bins, 3))  # 保存最近36帧的三通道历史数据

    def transform_pointcloud(self, pointcloud, quaternion, translation):
        """
        将点云从物体坐标系变换到世界坐标系

        参数:
            pointcloud: 输入点云
            quaternion: 四元数表示的旋转
            translation: 平移向量

        返回:
            变换后的世界坐标系下的点云
        """
        # 将四元数转换为旋转矩阵
        rot = R.from_quat(quaternion).as_matrix()
        # 应用旋转和平移变换
        return np.dot(pointcloud, rot.T) + translation

    def update_history(self, distance_vector):
        """
        更新历史数据

        参数:
            distance_vector: 当前帧的距离数据

        返回:
            history: 更新后的36帧历史数据
        """
        # 将最新数据滚动进入历史记录
        self.history = np.roll(self.history, -1, axis=0)
        # 替换最后列（最旧的数据）为新数据
        self.history[-1, :] = distance_vector
        return self.history

    def generate_obstacle_map(self, pointcloud, pose):
        """
        生成三通道障碍地图（RGB）
        修改：确保所有无人机都能显示，而不仅限于前3架，并优化亮度调节
        """
        # 初始化距离向量
        rock_vector = np.ones(self.n_bins)
        # 根据实际无人机数量初始化
        drone_vectors = {i: np.ones(self.n_bins) for i in range(self.num_drones)}
        
        if len(pointcloud) > 0:
            # 处理岩石点云（灰色通道）
            rock_mask = pointcloud[:, 3] < 0.5  # 反射率<0.5的是岩石
            if np.any(rock_mask):
                rock_vector = self.process_frame(pointcloud[rock_mask], pose)
            
            # 处理无人机点云（彩色通道）
            drone_mask = pointcloud[:, 3] >= 0.5  # 反射率>=0.5的是无人机
            if np.any(drone_mask):
                drone_points = pointcloud[drone_mask]
                # 按无人机编号处理（根据反射率值确定无人机编号）
                for i in range(self.num_drones):
                    # 根据反射率值确定对应无人机的点云
                    # 无人机i的反射率应该是0.8 + i * 0.025
                    expected_reflectance = 0.8 + i * 0.025
                    # 查找反射率接近预期值的点（允许一定误差）
                    drone_i_mask = np.abs(drone_points[:, 3] - expected_reflectance) < 0.01
                    if np.any(drone_i_mask):
                        drone_i_points = drone_points[drone_i_mask]
                        drone_vectors[i] = self.process_frame(drone_i_points, pose)
        
        # 分别更新各个通道的历史数据
        self.history = np.roll(self.history, -1, axis=0)
        
        # 替换最后列（最旧的数据）为新数据
        # 岩石使用相同值设置所有通道（灰色）
        self.history[-1, :, 0] = rock_vector  # 红色通道（灰度）
        self.history[-1, :, 1] = rock_vector  # 绿色通道（灰度）
        self.history[-1, :, 2] = rock_vector  # 蓝色通道（灰度）
        
        # 更新无人机通道，使用与右图一致的颜色表示
        # 通过RGB混合显示所有8架无人机
        # 右侧图颜色映射: 0-绿色, 1-蓝色, 2-红色, 3-橙色, 4-紫色, 5-棕色, 6-粉红, 7-青色
        for i in range(min(self.num_drones, 8)):  # 最多支持8架无人机
            if i in drone_vectors:
                if i == 0:  # 绿色 (0, 1, 0)
                    self.history[-1, :, 1] = np.minimum(self.history[-1, :, 1], drone_vectors[i])
                elif i == 1:  # 蓝色 (0, 0, 1)
                    self.history[-1, :, 2] = np.minimum(self.history[-1, :, 2], drone_vectors[i])
                elif i == 2:  # 红色 (1, 0, 0)
                    self.history[-1, :, 0] = np.minimum(self.history[-1, :, 0], drone_vectors[i])
                elif i == 3:  # 橙色 (1, 0.65, 0) - 近似为红+绿混合
                    self.history[-1, :, 0] = np.minimum(self.history[-1, :, 0], drone_vectors[i])
                    self.history[-1, :, 1] = np.minimum(self.history[-1, :, 1], 
                                                       np.maximum(0.65, drone_vectors[i]))  # 绿色分量为65%
                elif i == 4:  # 紫色 (0.5, 0, 0.5) - 近似为红+蓝混合
                    self.history[-1, :, 0] = np.minimum(self.history[-1, :, 0], 
                                                       np.maximum(0.5, drone_vectors[i]))  # 红色分量为50%
                    self.history[-1, :, 2] = np.minimum(self.history[-1, :, 2], 
                                                       np.maximum(0.5, drone_vectors[i]))  # 蓝色分量为50%
                elif i == 5:  # 棕色 - 确保与右侧图像一致
                    # 调整RGB分量以更好地表示棕色
                    self.history[-1, :, 0] = np.minimum(self.history[-1, :, 0], drone_vectors[i])
                    self.history[-1, :, 1] = np.minimum(self.history[-1, :, 1], 
                                                       np.maximum(0.36, drone_vectors[i]))  # 增加绿色分量以匹配brown颜色
                    self.history[-1, :, 2] = np.minimum(self.history[-1, :, 2], 
                                                       np.maximum(0.16, drone_vectors[i]))
                elif i == 6:  # 粉红 - 确保与右侧图像一致
                    # 调整RGB分量以更好地表示粉红色
                    self.history[-1, :, 0] = np.minimum(self.history[-1, :, 0], drone_vectors[i])  # 红色分量为主
                    self.history[-1, :, 1] = np.minimum(self.history[-1, :, 1], 
                                                       np.maximum(0.5, drone_vectors[i]))   # 降低绿色分量以突出粉红色
                    self.history[-1, :, 2] = np.minimum(self.history[-1, :, 2], 
                                                       np.maximum(0.7, drone_vectors[i]))   # 调整蓝色分量以匹配pink颜色
                elif i == 7:  # 青色 (0, 1, 1) - 绿+蓝通道，避免与灰色障碍物冲突
                    self.history[-1, :, 1] = np.minimum(self.history[-1, :, 1], drone_vectors[i])  # 绿色分量
                    self.history[-1, :, 2] = np.minimum(self.history[-1, :, 2], drone_vectors[i])  # 蓝色分量
        # 反转数值以便可视化（近处障碍更亮）
        # 使用gamma校正增强对比度，使近距离物体更明显
        gamma = 0.5  # gamma值小于1可以增强暗部细节
        obstacle_map_rgb = np.power(1 - self.history, gamma)
        # 确保数值不会超出范围
        obstacle_map_rgb = np.clip(obstacle_map_rgb, 0, 1)
        
        return obstacle_map_rgb  # 只返回RGB三通道数据


# 修正后的可视化函数
def visualize_environment_scene():
    """可视化环境场景和障碍地图"""
    # 使用默认障碍物位置
    env_obstacles = [(200, 320), (300, 100)]
    
    # 新增无人机参数（8架友机）
    simulator = EnvironmentObstacleSimulator(num_drones=8, h=3.0, env_obstacles=env_obstacles)  # 修改：指定h=3.0，8架无人机

    # 初始雷达姿态 (调整雷达位置使其更接近障碍物)
    pose = {
        'quaternion': [0, 0, 0, 1],
        'translation': [250, 200, 1.5]  # 调整位置使其更接近障碍物
    }

    final_pointcloud = np.array([])
    # 修改：无人机向右上角移动3.6秒，速度10m/s
    radar_direction = np.array([1, 1, 0])  # 向右上角移动（x和y方向）
    radar_direction = radar_direction / np.linalg.norm(radar_direction)  # 归一化方向向量
    radar_speed = 10.0  # 速度10m/s (修复速度值)
    total_time = 3.6  # 总时间3.6秒
    num_frames = 36  # 帧数
    time_per_frame = total_time / num_frames  # 每帧时间0.1秒
    
    # 计算每帧的位移
    displacement_per_frame = radar_direction * radar_speed * time_per_frame

    # 新增：存储雷达位置的历史记录
    radar_positions = []
    
    # 新增：为每个无人机存储轨迹点
    drone_trajectories = {i: [] for i in range(8)}  # 8架无人机轨迹历史
    
    # 初始化障碍物地图
    final_obstacle_map = None
    
    for i in range(num_frames):
        # 更新雷达位置
        pose['translation'] = np.array(pose['translation']) + displacement_per_frame

        # 将当前雷达位置加入历史记录
        radar_positions.append(pose['translation'].copy())

        # 处理当前帧数据
        obstacle_map, pointcloud = simulator.process_scene(pose)
        
        # 保存最后一帧的障碍物地图和点云数据
        if i == num_frames - 1:
            final_pointcloud = pointcloud
            final_obstacle_map = obstacle_map  # 保存最后一帧障碍地图
        
        # 新增：记录无人机位置到轨迹历史
        for drone_id, drone_points in enumerate(simulator.drone_pointclouds):
            if len(drone_points) > 0:
                drone_center = np.mean(drone_points[:, :3], axis=0)
                drone_trajectories[drone_id].append(drone_center.copy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))  # 增加图形宽度以适应图例

    # 障碍地图（RGBA格式，支持透明度）
    # 修复：确保显示正确的方向，角度在X轴，时间在Y轴
    # 修复图像维度以正确显示竖直线
    if final_obstacle_map is not None:
        ax1.imshow(final_obstacle_map, origin='lower', vmin=0, vmax=1)
        # 修改标题说明透明度含义
        ax1.set_title('RGBA Obstacle Map\nGray: Obstacles, Colored: Drones\nBrightness = Proximity, Alpha = Distance')
        ax1.set_xlabel('Angle Bins (36 bins)')  # 角度分箱
        ax1.set_ylabel('Time Frames (36 frames)')  # 时间帧数

        # 手动添加白色网格线（横竖各35条）
        for i in range(1, 36):
            ax1.axvline(i - 0.5, color='white', linestyle='-', linewidth=0.5)
            ax1.axhline(i - 0.5, color='white', linestyle='-', linewidth=0.5)
    else:
        ax1.text(0.5, 0.5, 'No Obstacle Map', ha='center', va='center')
        ax1.set_title('Obstacle Map')

    # 点云二维投影（XY平面）
    if len(final_pointcloud) > 0:
        # 修改点云坐标转换：仅对前3列坐标进行平移，保留反射率属性
        coords = final_pointcloud[:, :3] + radar_positions[-1]  # 仅坐标部分平移
        reflectances = final_pointcloud[:, 3]  # 保留反射率列
        world_pointcloud = np.column_stack((coords, reflectances))  # 重新组合
        
        # 点云使用世界坐标绘制
        ax2.scatter(world_pointcloud[:, 0], world_pointcloud[:, 1], s=1, c='black')

        # 绘制雷达的运动路径
        radar_x = [pos[0] for pos in radar_positions]
        radar_y = [pos[1] for pos in radar_positions]
        ax2.plot(radar_x, radar_y, color='red', linestyle='--', label='Radar Path')

        # 在雷达位置处添加朝向指示器
        ax2.plot([radar_positions[-1][0], radar_positions[-1][0] + 1],
                 [radar_positions[-1][1], radar_positions[-1][1]],
                 color='blue', linewidth=2)
        # , label = 'Radar Facing'

        colors = ['green', 'purple', 'orange', 'cyan', 'magenta']  # 不同障碍物使用不同颜色

        # 修改：直接使用模拟器中存储的障碍物点云数据
        for obstacle_id, obstacle_points in enumerate(simulator.obstacle_pointclouds):
            if len(obstacle_points) > 0:
                # 直接使用世界坐标计算中心（关键修改）
                obstacle_center = np.mean(obstacle_points, axis=0)
                color = colors[obstacle_id % len(colors)]

                # 绘制连接线（使用世界坐标）
                ax2.plot([radar_positions[-1][0], obstacle_center[0]],
                         [radar_positions[-1][1], obstacle_center[1]],
                         color=color, linewidth=1)
                # label = f'Obstacle {obstacle_id + 1} Angle'

                # 角度计算（世界坐标系）修改为统一计算方式
                dx = obstacle_center[0] - radar_positions[-1][0]
                dy = obstacle_center[1] - radar_positions[-1][1]
                # 修正：使用 arctan2(dy, dx) 确保与障碍地图计算一致
                angle = np.rad2deg(np.arctan2(dy, dx)) % 360
                ax2.text(obstacle_center[0], obstacle_center[1],
                         f'{angle:.1f}°',
                         fontsize=8, color=color,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 绘制每个无人机的轨迹，使用与左侧障碍物地图一致的颜色表示方法
        for drone_id in range(simulator.num_drones):
            if drone_id in drone_trajectories and len(drone_trajectories[drone_id]) > 0:
                # 获取该无人机的轨迹点
                trajectory = np.array(drone_trajectories[drone_id])
                
                # 根据无人机ID选择与左侧障碍物地图一致的颜色
                if drone_id == 0:  # 绿色
                    color = 'green'
                elif drone_id == 1:  # 蓝色
                    color = 'blue'
                elif drone_id == 2:  # 红色
                    color = 'red'
                elif drone_id == 3:  # 橙色
                    color = 'orange'
                elif drone_id == 4:  # 紫色
                    color = 'purple'
                elif drone_id == 5:  # 棕色
                    color = 'brown'
                elif drone_id == 6:  # 粉红
                    color = 'pink'
                elif drone_id == 7:  # 青色
                    color = 'cyan'
                else:
                    color = 'black'  # 默认黑色
                
                # 绘制轨迹线
                ax2.plot(trajectory[:, 0], trajectory[:, 1], 
                         color=color, linestyle='-', linewidth=2,
                         label=f'Drone{drone_id} Trajectory')
        
        # 标注无人机反射率和ID，使用与左侧障碍物地图一致的颜色表示方法
        for drone_id, drone_points in enumerate(simulator.drone_pointclouds):
            if len(drone_points) > 0:
                # 无人机中心位置（前三列）
                drone_center = np.mean(drone_points[:, :3], axis=0)
                
                # 根据无人机ID选择与左侧障碍物地图一致的颜色
                if drone_id == 0:  # 绿色
                    color = 'green'
                elif drone_id == 1:  # 蓝色
                    color = 'blue'
                elif drone_id == 2:  # 红色
                    color = 'red'
                elif drone_id == 3:  # 橙色
                    color = 'orange'
                elif drone_id == 4:  # 紫色
                    color = 'purple'
                elif drone_id == 5:  # 棕色
                    color = 'brown'
                elif drone_id == 6:  # 粉红
                    color = 'pink'
                elif drone_id == 7:  # 青色
                    color = 'cyan'
                else:
                    color = 'black'  # 默认黑色
                
                # 角度计算
                dx = drone_center[0] - radar_positions[-1][0]
                dy = drone_center[1] - radar_positions[-1][1]
                angle = np.rad2deg(np.arctan2(dy, dx)) % 360
                
                # 标注无人机反射率、ID和到本机的距离
                avg_reflectance = np.mean(drone_points[:, 3])
                # 计算到本机无人机的距离
                distance = np.sqrt((drone_center[0] - radar_positions[-1][0])**2 + 
                                 (drone_center[1] - radar_positions[-1][1])**2)
                ax2.text(drone_center[0], drone_center[1] + 0.5,
                         f'Drone{drone_id}: {angle:.1f}°\nReflect: {avg_reflectance:.2f}\nDistance: {distance:.2f}m',
                         fontsize=8, color=color,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 修改：将图例显示在图外边
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_title('Point Cloud Projection (XY Plane)')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        # 修复：使用set_aspect确保等比例显示，避免使用axis('equal')可能引起的冲突
        ax2.set_aspect('equal', adjustable='box')
    else:
        ax2.text(0.5, 0.5, 'No Points', ha='center', va='center')
        ax2.set_title('Point Cloud Projection (XY Plane)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_environment_scene()