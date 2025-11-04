import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R  # 用于四元数到旋转矩阵的转换
import random
matplotlib.use('TkAgg')


class RockObstacleSimulator:
    """模拟岩石障碍物点云生成器"""
    
    def __init__(self, n_bins=36, d_max=10.0, h=3.0, num_rocks=3, num_drones=1):
        """
        初始化模拟器
        修改：增大垂直检测范围h=3.0以覆盖无人机高度
        """
        self.lidar_encoder = LidarEncoder(n_bins, d_max, h)  # 使用新的h值
        self.num_rocks = num_rocks  # 岩石数量
        self.num_drones = num_drones  # 无人机数量
        self.rock_params = []  # 存储每个岩石的参数
        self.drone_params = []  # 存储每个无人机的参数（新增）
        # 新增：为每个无人机分配唯一颜色
        self.drone_colors = ['green', 'blue']  # 无人机0:绿色, 无人机1:蓝色

    def generate_rock_pointcloud(self, rock_id, pose):
        """
        生成单个岩石的点云数据

        参数:
            rock_id: 岩石的唯一标识符
            pose: LiDAR传感器的姿态信息，包含四元数和位移

        返回:
            rock_points: 相对于LiDAR坐标系的岩石点云数据
        """
        # 如果是新的岩石ID，则随机生成其参数
        if rock_id >= len(self.rock_params):
            x = random.uniform(-8, 8)  # 随机x坐标
            y = random.uniform(-8, 8)  # 随机y坐标
            z = random.uniform(0.2, 0.8)  # 随机z坐标
            radius = random.uniform(0.5, 1.5)  # 随机半径
            point_density = random.randint(30, 80)  # 随机点密度

            self.rock_params.append({
                'pos': np.array([x, y, z]),  # 岩石中心位置
                'radius': radius,  # 岩石半径
                'density': point_density  # 点密度
            })

        # 获取当前岩石参数
        params = self.rock_params[rock_id]
        num_points = params['density']  # 点数量
        center = params['pos']  # 中心位置
        radius = params['radius']  # 半径

        # 在球体内生成随机点（使用球面坐标）
        r = np.random.normal(0, radius / 3, num_points)  # 径向距离（加入噪声）
        theta = np.random.uniform(0, 2 * np.pi, num_points)  # 方位角
        phi = np.random.uniform(0, np.pi, num_points)  # 极角

        # 转换为笛卡尔坐标
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        # 过滤掉超出实际半径的点
        valid = np.sqrt(x ** 2 + y ** 2 + z ** 2) <= radius
        x, y, z = x[valid], y[valid], z[valid]

        # 根据与中心的距离调整点密度（越靠近表面点越密集）
        density_factor = 1 - np.sqrt(x ** 2 + y ** 2 + z ** 2) / radius
        keep_mask = np.random.random(len(x)) < density_factor
        x, y, z = x[keep_mask], y[keep_mask], z[keep_mask]

        # 生成世界坐标点云
        world_points = np.column_stack((x, y, z)) + center
        return world_points  # 直接返回世界坐标

    def generate_drone_pointcloud(self, drone_id, pose):
        """
        生成无人机友机的点云数据
        特点：
        - 立方体形状（模拟无人机）
        - 高反射率属性（模拟LED）
        
        参数:
            drone_id: 无人机的唯一标识符
            pose: LiDAR传感器的姿态信息
            
        返回:
            drone_points: 带反射率属性的无人机点云
        """
        # 如果是新的无人机ID，则随机生成其参数
        if drone_id >= len(self.drone_params):
            x = random.uniform(-8, 8)  # 随机x坐标
            y = random.uniform(-8, 8)  # 随机y坐标
            z = random.uniform(2, 5)   # 无人机飞行高度
            size = random.uniform(0.3, 0.8)  # 无人机尺寸
            # LED反射率属性（0.8-1.0表示高反射）
            reflectance = random.uniform(0.8, 1.0)  

            self.drone_params.append({
                'pos': np.array([x, y, z]),
                'size': size,
                'reflectance': reflectance  # 反射率属性
            })

        # 获取当前无人机参数
        params = self.drone_params[drone_id]
        center = params['pos']
        size = params['size']
        reflectance = params['reflectance']  # 反射率值

        # 生成立方体点云（模拟无人机形状）
        num_points = 50  # 点数量
        half_size = size / 2
        
        # 生成立方体表面的点
        x = np.random.uniform(-half_size, half_size, num_points) + center[0]
        y = np.random.uniform(-half_size, half_size, num_points) + center[1]
        z = np.random.uniform(-half_size, half_size, num_points) + center[2]
        
        # 添加反射率属性（第四列）
        intensities = np.full(num_points, reflectance)
        drone_points = np.column_stack((x, y, z, intensities))
        
        return drone_points

    def generate_scene_pointcloud(self, pose):
        """
        生成整个场景的点云数据（包含岩石和无人机）
        """
        lidar_pos = np.array(pose['translation'])
        all_points = []
        self.rock_pointclouds = []  
        self.drone_pointclouds = []  # 存储无人机点云（新增）

        # 生成岩石点云
        for i in range(self.num_rocks):
            rock_points = self.generate_rock_pointcloud(i, pose)
            # 为岩石添加反射率属性（0.1-0.3表示低反射）
            rock_intensity = np.full(rock_points.shape[0], random.uniform(0.1, 0.3))
            rock_points = np.column_stack((rock_points, rock_intensity))
            
            all_points.append(rock_points)
            self.rock_pointclouds.append(rock_points)

        # 生成无人机点云（新增）
        for i in range(self.num_drones):
            drone_points = self.generate_drone_pointcloud(i, pose)
            all_points.append(drone_points)
            self.drone_pointclouds.append(drone_points)

        # 转换为LiDAR坐标系
        lidar_points = [points[:, :3] - lidar_pos for points in all_points]  # 只转换坐标
        # 保留反射率属性（第四列）
        intensities = [points[:, 3] for points in all_points]
        
        # 合并坐标和反射率
        merged_points = []
        for coords, intensity in zip(lidar_points, intensities):
            merged_points.append(np.column_stack((coords, intensity)))
            
        return np.vstack(merged_points) if merged_points else np.array([])

    def process_rock_scene(self, pose):
        """
        处理整个岩石场景，生成障碍地图和点云数据

        参数:
            pose: LiDAR传感器的姿态信息

        返回:
            obstacle_map: 障碍地图
            rock_pointcloud: 场景点云数据
        """
        # 生成整个场景的点云
        rock_pointcloud = self.generate_scene_pointcloud(pose)

        # 新增：计算并输出点云相对于雷达的平均角度
        if len(rock_pointcloud) > 0:
            # 修正：只对坐标部分(前3列)进行减法运算，保留反射率属性
            coords = rock_pointcloud[:, :3]  # 提取坐标部分
            rel_vectors = coords - np.array(pose['translation'])  # 仅坐标部分减去雷达位置
            
            # 计算角度（arctan2(y,x)）
            angles = np.arctan2(rel_vectors[:, 1], rel_vectors[:, 0])
            # 修改：将角度转换为0-360度范围
            angles_deg = np.rad2deg(angles) % 360

            # 计算雷达正方向角度（从四元数）
            quat = pose['quaternion']
            rot_matrix = R.from_quat(quat).as_matrix()
            radar_direction = rot_matrix[:, 0]  # 获取X轴方向（雷达正方向）
            # 修改：转换为0-360度格式
            radar_angle = np.rad2deg(np.arctan2(radar_direction[1], radar_direction[0])) % 360

            # print(f"[DEBUG] 雷达正方向角度: {radar_angle:.2f}°")
            # print(f"[DEBUG] 石头点云相对于雷达的平均角度: {np.mean(angles_deg):.2f}°")

        # 使用LiDAR编码器生成障碍地图
        obstacle_map = self.lidar_encoder.generate_obstacle_map(rock_pointcloud, pose)
        return obstacle_map, rock_pointcloud


class LidarEncoder:
    """
    LiDAR 数据编码器
    功能：
    - 处理LiDAR点云数据
    - 生成障碍地图
    - 维护历史数据以显示时间序列信息
    """

    def __init__(self, n_bins=36, d_max=10.0, h=1.0):
        """
        初始化LiDAR编码器

        参数:
            n_bins: 角度分箱数（36表示每10度一个分箱）
            d_max: 最大检测距离
            h: LiDAR的垂直检测范围（从中心上下各h米）
        """
        self.n_bins = n_bins  # 角度分箱数
        self.d_max = d_max  # 最大检测距离
        self.h = h  # 垂直检测范围
        # 修改：历史数据扩展为三通道（RGB）
        self.history = np.ones((n_bins, 36, 3))  # 保存最近36帧的三通道历史数据
        # 新增：存储无人机历史位置
        self.drone_history = {0: [], 1: []}  # 最多支持两个无人机

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

    def process_frame(self, pointcloud, pose):
        """
        处理单帧LiDAR数据
        修改：放宽垂直检测范围过滤条件
        """
        # 获取姿态信息
        quaternion = pose['quaternion']
        translation = pose['translation']
        
        # 分离坐标和反射率（新增）
        coords = pointcloud[:, :3]  # 前三列为坐标
        intensities = pointcloud[:, 3] if pointcloud.shape[1] > 3 else None  # 第四列为反射率
        
        # 将点云转换到世界坐标系（只转换坐标）
        world_points = self.transform_pointcloud(coords, quaternion, translation)

        # 计算LiDAR所在平面的z轴范围
        z_q = translation[2]
        z_min, z_max = z_q - self.h, z_q + self.h

        # 修改：放宽垂直检测范围过滤条件，允许无人机点云通过
        # 仅对岩石点云进行严格的高度过滤
        rock_mask = pointcloud[:, 3] < 0.5 if pointcloud.shape[1] > 3 else None
        if rock_mask is not None:
            valid_rock = np.logical_and(world_points[:, 2] >= z_min, 
                                        world_points[:, 2] <= z_max)
            valid_rock = np.logical_and(valid_rock, rock_mask)
        else:
            valid_rock = np.logical_and(world_points[:, 2] >= z_min, 
                                        world_points[:, 2] <= z_max)
        
        # 无人机点云不进行高度过滤
        drone_mask = pointcloud[:, 3] >= 0.5 if pointcloud.shape[1] > 3 else None
        valid_drone = drone_mask if drone_mask is not None else np.ones(len(world_points), dtype=bool)
        
        # 合并有效点
        valid = np.logical_or(valid_rock, valid_drone)
        filtered_coords = world_points[valid]
        # 同时过滤反射率（新增）
        filtered_intensities = intensities[valid] if intensities is not None else None

        # 如果没有有效点，返回全1向量（表示无障碍）
        if len(filtered_coords) == 0:
            return np.ones(self.n_bins)

        # 转换到LiDAR局部坐标系
        rel_points = filtered_coords - translation
        # 计算每个点到LiDAR的距离
        ranges = np.linalg.norm(rel_points, axis=1)
        # 角度计算：使用标准的arctan2(y,x)计算，范围[0, 2π]
        angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
        # 将角度转换为[0, 2π]范围
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)

        # 角度分箱范围改为[0, 2π]
        angle_bins = np.linspace(0, 2 * np.pi, self.n_bins + 1)
        # 确定每个点属于哪个角度分箱
        bin_indices = np.digitize(angles, angle_bins) - 1

        # 修正：处理边界值(2π)的特殊情况
        # 当角度等于2π时，np.digitize会返回n_bins+1，减去1后为n_bins，应修正为0
        bin_indices[bin_indices == self.n_bins] = 0

        # 确保索引在[0, n_bins-1]范围内
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # 初始化每个角度方向的最大距离为最大检测距离
        distance_vector = np.ones(self.n_bins) * self.d_max
        # 对于每个角度分箱，保留最小距离的点
        for i, idx in enumerate(bin_indices):
            if ranges[i] < distance_vector[idx]:
                distance_vector[idx] = ranges[i]

        # 归一化距离值
        return distance_vector / self.d_max

    def update_history(self, distance_vector):
        """
        更新历史数据

        参数:
            distance_vector: 当前帧的距离数据

        返回:
            history: 更新后的36帧历史数据
        """
        # 将最新数据滚动进入历史记录
        self.history = np.roll(self.history, -1, axis=1)
        # 替换最后列（最旧的数据）为新数据
        self.history[:, -1] = distance_vector
        return self.history

    def generate_obstacle_map(self, pointcloud, pose, drone_positions=None):
        """
        生成三通道障碍地图（RGB）
        修改：确保无人机通道不被岩石通道覆盖
        """
        # 初始化三个通道的距离向量
        rock_vector = np.ones(self.n_bins)
        drone_vectors = {0: np.ones(self.n_bins), 1: np.ones(self.n_bins)}
        
        if len(pointcloud) > 0:
            # 处理岩石点云（灰色通道）
            rock_mask = pointcloud[:, 3] < 0.5  # 反射率<0.5的是岩石
            if np.any(rock_mask):
                rock_vector = self.process_frame(pointcloud[rock_mask], pose)
            
            # 处理无人机点云（绿色/蓝色通道）
            drone_mask = pointcloud[:, 3] >= 0.5  # 反射率>=0.5的是无人机
            if np.any(drone_mask):
                drone_points = pointcloud[drone_mask]
                # 按反射率分组无人机
                for i in range(min(len(self.drone_history), 2)):  # 最多两个无人机
                    # 计算当前帧的无人机位置
                    drone_pos = np.mean(drone_points[:, :3], axis=0) if len(drone_points) > 0 else None
                    if drone_pos is not None:
                        # 更新无人机历史位置
                        self.drone_history[i].append(drone_pos.copy())
                        if len(self.drone_history[i]) > 5:  # 保留最近5帧
                            self.drone_history[i] = self.drone_history[i][-5:]
                        
                        # 计算运动趋势（最近3帧的平均位移）
                        if len(self.drone_history[i]) >= 3:
                            recent_pos = np.array(self.drone_history[i][-3:])
                            movement = np.mean(np.diff(recent_pos, axis=0), axis=0)
                        else:
                            movement = np.zeros(3)
                        
                        # 生成带运动趋势的点云
                        trend_points = np.copy(drone_points)
                        trend_points[:, :3] += movement * 2  # 放大运动趋势
                        drone_vectors[i] = self.process_frame(trend_points, pose)
        
        # 分别更新三个通道的历史数据
        self.history[:, :, 0] = np.roll(self.history[:, :, 0], -1, axis=1)
        self.history[:, :, 1] = np.roll(self.history[:, :, 1], -1, axis=1)
        self.history[:, :, 2] = np.roll(self.history[:, :, 2], -1, axis=1)
        
        # 替换最后列（最旧的数据）为新数据
        # 岩石使用相同值设置三通道（灰色）
        self.history[:, -1, 0] = rock_vector
        
        # 修改：无人机通道独立设置，不覆盖岩石值
        # 使用无人机自身距离值，避免被岩石值覆盖
        # 无人机0（绿色通道）
        self.history[:, -1, 1] = drone_vectors[0]
        # 无人机1（蓝色通道）
        self.history[:, -1, 2] = drone_vectors[1]
        
        # 反转数值以便可视化（近处障碍更亮）
        # 修改：使用距离值计算透明度（近处透明度低，远处高）
        # 计算每个位置的最小距离（三个通道中的最小值）
        min_distance = np.min(self.history, axis=2)
        # 计算透明度因子：alpha = 1 - (1 - min_distance)^gamma (gamma控制对比度)
        gamma = 0.5  # 对比度调整因子
        alpha_factor = 1 - (1 - min_distance) ** gamma
        
        # 将RGB值与透明度结合 (RGBA格式)
        obstacle_map_rgb = (1 - self.history) * 255
        obstacle_map_alpha = (alpha_factor * 255).astype(np.uint8)  # Alpha通道
        
        # 合并为四通道图像 (RGBA)
        obstacle_map = np.zeros((self.n_bins, 36, 4), dtype=np.uint8)
        obstacle_map[..., :3] = obstacle_map_rgb.astype(np.uint8)
        obstacle_map[..., 3] = obstacle_map_alpha  # 使用二维Alpha数组
        
        return obstacle_map


# 修正后的可视化函数
def visualize_rock_scene():
    """可视化岩石场景和障碍地图"""
    # 新增无人机参数
    simulator = RockObstacleSimulator(num_rocks=2, num_drones=1, h=3.0)  # 修改：指定h=3.0

    # 初始雷达姿态
    pose = {
        'quaternion': [0, 0, 0, 1],
        'translation': [0, 0, 1.5]
    }

    final_pointcloud = np.array([])
    radar_direction = np.array([1, 0, 0])  # 雷达移动的方向（沿X轴正方向）
    radar_speed = 0.1  # 雷达移动的速度

    # 新增：存储雷达位置的历史记录
    radar_positions = []
    
    # 新增：为每个无人机存储轨迹点
    drone_trajectories = {0: [], 1: []}  # 无人机轨迹历史
    
    for i in range(36):
        # 更新雷达位置
        pose['translation'] = np.array(pose['translation']) + radar_direction * radar_speed

        # 将当前雷达位置加入历史记录
        radar_positions.append(pose['translation'].copy())

        # 处理当前帧数据
        obstacle_map, pointcloud = simulator.process_rock_scene(pose)
        
        if i == 35:
            final_pointcloud = pointcloud
            final_obstacle_map = obstacle_map  # 保存最后一帧障碍地图
        
        # 新增：记录无人机位置到轨迹历史
        for drone_id, drone_points in enumerate(simulator.drone_pointclouds):
            if len(drone_points) > 0:
                drone_center = np.mean(drone_points[:, :3], axis=0)
                drone_trajectories[drone_id].append(drone_center.copy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 障碍地图（RGBA格式，支持透明度）
    ax1.imshow(final_obstacle_map.transpose(1, 0, 2), vmin=0, vmax=255)
    # 修改标题说明透明度含义
    ax1.set_title('RGBA Obstacle Map\nGray: Rocks, Green: Drone0, Blue: Drone1\nBrightness = Proximity, Alpha = Distance')
    ax1.set_xlabel('Angle Bins (36 bins)')
    ax1.set_ylabel('Time Frames (36 frames)')

    # 手动添加白色网格线（横竖各35条）
    for i in range(1, 36):
        ax1.axvline(i - 0.5, color='white', linestyle='-', linewidth=0.5)
        ax1.axhline(i - 0.5, color='white', linestyle='-', linewidth=0.5)

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

        colors = ['green', 'purple', 'orange', 'cyan', 'magenta']  # 不同岩石使用不同颜色

        # 修改：直接使用模拟器中存储的岩石点云数据
        for rock_id, rock_points in enumerate(simulator.rock_pointclouds):
            if len(rock_points) > 0:
                # 直接使用世界坐标计算中心（关键修改）
                rock_center = np.mean(rock_points, axis=0)
                color = colors[rock_id % len(colors)]

                # 绘制连接线（使用世界坐标）
                ax2.plot([radar_positions[-1][0], rock_center[0]],
                         [radar_positions[-1][1], rock_center[1]],
                         color=color, linewidth=1)
                # label = f'Rock {rock_id + 1} Angle'

                # 角度计算（世界坐标系）修改为统一计算方式
                dx = rock_center[0] - radar_positions[-1][0]
                dy = rock_center[1] - radar_positions[-1][1]
                # 修正：使用 arctan2(dy, dx) 确保与障碍地图计算一致
                angle = np.rad2deg(np.arctan2(dy, dx)) % 360
                ax2.text(rock_center[0], rock_center[1],
                         f'{angle:.1f}°',
                         fontsize=8, color=color,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 新增：无人机颜色映射表
        drone_colors = ['green', 'blue']  # 无人机0:绿色, 无人机1:蓝色
        
        # 新增：绘制每个无人机的轨迹
        for drone_id in range(simulator.num_drones):
            if drone_id in drone_trajectories and len(drone_trajectories[drone_id]) > 0:
                # 获取该无人机的轨迹点和颜色
                trajectory = np.array(drone_trajectories[drone_id])
                color = simulator.drone_colors[drone_id] if drone_id < len(simulator.drone_colors) else 'purple'
                
                # 绘制轨迹线
                ax2.plot(trajectory[:, 0], trajectory[:, 1], 
                         color=color, linestyle='-', linewidth=2,
                         label=f'Drone{drone_id} Trajectory')
        
        # 新增：标注无人机反射率和ID
        for drone_id, drone_points in enumerate(simulator.drone_pointclouds):
            if len(drone_points) > 0:
                # 无人机中心位置（前三列）
                drone_center = np.mean(drone_points[:, :3], axis=0)
                # 根据ID选择颜色
                color = drone_colors[drone_id] if drone_id < len(drone_colors) else 'purple'
                
                # 角度计算
                dx = drone_center[0] - radar_positions[-1][0]
                dy = drone_center[1] - radar_positions[-1][1]
                angle = np.rad2deg(np.arctan2(dy, dx)) % 360
                
                # 标注无人机反射率和ID
                avg_reflectance = np.mean(drone_points[:, 3])
                ax2.text(drone_center[0], drone_center[1] + 0.5,
                         f'Drone{drone_id}: {angle:.1f}°\nReflect: {avg_reflectance:.2f}',
                         fontsize=8, color=color,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax2.legend(loc='upper right')
        ax2.set_title('Point Cloud Projection (XY Plane)')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.axis('equal')
    else:
        ax2.text(0.5, 0.5, 'No Points', ha='center', va='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_rock_scene()