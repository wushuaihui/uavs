import numpy as np
import random
import copy
from env import Point2D

class GWO:
    """
    灰狼优化算法(Grey Wolf Optimizer)实现
    用于拒止区域内的无人机跟随路径规划
    """
    
    def __init__(self, pop_size, dim, ub, lb, max_iter):
        """
        初始化GWO算法参数
        
        参数:
        pop_size: 种群大小（灰狼数量）
        dim: 问题维度（此处为2，代表x,y坐标）
        ub: 搜索空间上界
        lb: 搜索空间下界
        max_iter: 最大迭代次数
        """
        self.pop_size = pop_size
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.max_iter = max_iter
        
        # 初始化灰狼种群位置 Alpha, Beta, Delta为前三只狼
        self.positions = np.random.uniform(lb, ub, (pop_size, dim))
        
        # 初始化Alpha, Beta, Delta狼的适应度值（初始化为无穷大）
        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")
        
        # 初始化Alpha, Beta, Delta狼的位置
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        
    def optimize(self, target_positions, own_position):
        """
        执行GWO优化过程
        
        参数:
        target_positions: 可跟随目标的位置列表 [(x,y), ...]，这些是相对于当前无人机雷达的局部坐标
        own_position: 当前无人机位置 (x, y)，在局部坐标系中始终为(0, 0)
        
        返回:
        best_position: 最优跟随位置 (x, y)
        """
        # 如果没有可跟随的目标，返回当前位置
        if not target_positions:
            return own_position
            
        # 迭代优化过程
        for iter in range(self.max_iter):
            for i in range(self.pop_size):
                # 计算当前狼的适应度
                fitness = self.calculate_fitness(
                    self.positions[i], 
                    target_positions, 
                    own_position  # 在局部坐标系中为(0, 0)
                )
                
                # 更新Alpha, Beta, Delta狼
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = copy.deepcopy(self.positions[i])
                    
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = copy.deepcopy(self.positions[i])
                    
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = copy.deepcopy(self.positions[i])
            
            # 计算系数a（线性递减从2到0）
            a = 2 - iter * (2 / self.max_iter)
            
            # 更新狼群位置
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # Alpha狼影响
                    r1 = random.random()
                    r2 = random.random()
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Beta狼影响
                    r1 = random.random()
                    r2 = random.random()
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Delta狼影响
                    r1 = random.random()
                    r2 = random.random()
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # 更新位置
                    self.positions[i, j] = (X1 + X2 + X3) / 3
            
            # 确保狼群位置在边界内
            self.positions = np.clip(self.positions, self.lb, self.ub)
        
        # 返回最优位置（Alpha狼位置）
        return self.alpha_pos
    
    def calculate_fitness(self, position, target_positions, own_position):
        """
        计算位置适应度值（改进版本）
        
        参数:
        position: 待评估位置 [x, y]
        target_positions: 可跟随目标位置列表，这些是相对于当前无人机雷达的局部坐标
        own_position: 当前无人机位置
        
        返回:
        fitness: 适应度值（越小越好）
        """
        x, y = position  # 当前待评估的位置（局部坐标）
        
        # 初始化适应度值
        fitness = 0
        
        # 1. 计算到最近目标的距离奖励（优化距离范围）
        min_target_dist = float('inf')
        for target_x, target_y in target_positions:
            # 在局部坐标系中计算距离
            dist = np.sqrt((x - target_x)**2 + (y - target_y)**2)
            if dist < min_target_dist:
                min_target_dist = dist
        
        # 优化距离奖励：鼓励保持15-25单位的理想跟随距离
        if min_target_dist < 15:
            # 距离过近，给予惩罚
            fitness += (15 - min_target_dist) * 2
        elif min_target_dist > 25:
            # 距离过远，给予惩罚
            fitness += (min_target_dist - 25) * 1.5
        else:
            # 理想距离范围，给予奖励
            fitness -= 50 / (1 + min_target_dist)
        
        # 2. 与当前位置距离惩罚（避免移动过大，优化惩罚系数）
        # own_position是世界坐标，但在局部坐标系中当前位置是(0,0)
        move_dist = np.sqrt((x - 0)**2 + (y - 0)**2)
        # 优化移动惩罚：鼓励小幅度平稳移动
        if move_dist > 20:
            fitness += move_dist * 0.2  # 大幅移动惩罚
        else:
            fitness += move_dist * 0.05  # 小幅移动轻微惩罚
        
        # 3. 群体聚集奖励（如果有多个目标，鼓励向群体中心移动）
        if len(target_positions) > 1:
            center_x = sum([pos[0] for pos in target_positions]) / len(target_positions)
            center_y = sum([pos[1] for pos in target_positions]) / len(target_positions)
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # 优化群体聚集奖励：鼓励保持在群体中心附近
            if dist_to_center < 30:
                fitness -= 40 / (1 + dist_to_center)  # 靠近中心奖励
            else:
                fitness += dist_to_center * 0.1  # 远离中心惩罚
            
        # 4. 保持适中编队距离奖励（避免过于分散或过于紧密）
        if len(target_positions) > 0:
            # 计算到所有目标的平均距离
            distances = [np.sqrt((x - target_x)**2 + (y - target_y)**2) 
                        for target_x, target_y in target_positions]
            avg_dist = np.mean(distances)
            
            # 优化编队距离：鼓励保持15-25单位的理想距离
            if 15 <= avg_dist <= 25:
                fitness -= 25  # 理想距离给予较大奖励
            elif avg_dist < 10:
                fitness += (10 - avg_dist) * 3  # 过近距离惩罚
            elif avg_dist > 40:
                fitness += (avg_dist - 40) * 2  # 过远距离惩罚
            else:
                fitness += abs(avg_dist - 20) * 0.5  # 轻微惩罚
        
        # 5. 路径平滑性奖励（鼓励选择平滑的移动方向）
        if len(target_positions) > 0:
            # 计算当前位置与目标位置连线的角度一致性
            angles = []
            for target_x, target_y in target_positions:
                angle = np.arctan2(target_y - y, target_x - x)
                angles.append(angle)
            
            # 计算角度方差，方差越小表示方向越一致
            if len(angles) > 1:
                angle_variance = np.var(angles)
                fitness -= 20 / (1 + angle_variance)  # 方向一致奖励
        
        return fitness


def gwo_follow(target_positions, own_position, 
               pop_size=15, max_iter=30, search_range=50):
    """
    使用GWO算法计算最优跟随位置
    
    参数:
    target_positions: 检测到的目标位置列表 [(x,y), ...]，这些是相对于当前无人机雷达的局部坐标
    own_position: 当前无人机位置 (x, y)，这是基于惯性导航或最后已知位置的估算值
    pop_size: GWO种群大小（优化为15，减少计算量）
    max_iter: 最大迭代次数（优化为30，提高收敛速度）
    search_range: 搜索范围（优化为50，减少路径震荡）
    
    返回:
    best_position: 最优跟随位置 (x, y)，这是相对于雷达坐标的局部位置
    """
    
    # 如果没有可跟随的目标，返回当前位置
    if not target_positions:
        return own_position
    
    # 如果只有一个目标，直接跟随该目标
    if len(target_positions) == 1:
        # 保持适中的跟随距离（10-20单位）
        target = target_positions[0]
        # 在局部坐标系中，当前位置是(0,0)
        dx = target[0] - 0  
        dy = target[1] - 0
        distance = np.sqrt(dx*dx + dy*dy)
        
        # 如果距离适中，直接返回目标位置
        if 10 <= distance <= 20:
            return target
        else:
            # 计算理想跟随位置（保持15单位距离）
            ideal_distance = 15
            if distance > 0:
                # 按比例缩放以达到理想距离
                scale = ideal_distance / distance
                # 返回局部坐标系中的位置
                return [0 + dx * scale, 0 + dy * scale]
            else:
                return target
    
    # 定义搜索边界（以当前位置为中心的正方形区域）
    # 在拒止区域内，我们使用局部雷达坐标系统
    # 缩小搜索范围以减少路径震荡
    lb = [-search_range, -search_range]  # 下界（相对于当前位置的局部坐标）
    ub = [search_range, search_range]  # 上界（相对于当前位置的局部坐标）
    
    # 创建GWO优化器实例
    gwo = GWO(pop_size, 2, ub, lb, max_iter)
    
    # 执行优化，当前位置在局部坐标系中为(0,0)
    best_position = gwo.optimize(target_positions, [0, 0])
    
    # 返回局部坐标，不需要转换为全局坐标
    # 在拒止区域内，控制器需要的是相对于当前位置的局部坐标来计算动作
    return best_position
