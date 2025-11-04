import os
import csv
import numpy as np
from matplotlib import pyplot as plt


def drawback():
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axis([0, 500, 0, 500])


def drawtarget(x,y):
    plt.text(x - 18, y - 30, 'Target')
    r = 20.0
    # 2.圆心坐标
    a = x
    b = y
    # ==========================================
    # 方法一：参数方程
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    plt.plot(x, y,linestyle='solid',c = 'black')
    # plt.legend(fontsize=10, loc="lower right")

def add_legend():
    # 确保所有标签都已被注册
    plt.legend(fontsize=10, loc="lower right")

def save_lidar_image(lidar_image, episode, drone_id):
    """
    保存激光雷达图像，确保与输入DRL的图像完全一致
    :param lidar_image: 激光雷达图像数据（应该与getLidarObservations返回的图像一致）
    :param episode: 当前episode
    :param drone_id: 无人机ID
    """
    figure_save_path = "sac1_results/lidar_images"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    
    # 确保图像数据在[0,1]范围内，与输入DRL的图像保持一致
    lidar_image = np.clip(lidar_image, 0, 1)
    
    # 保存激光雷达图像
    # 如果是RGB图像（3通道），则使用默认的RGB色彩映射
    # 如果是灰度图像（单通道），则使用灰度色彩映射
    if len(lidar_image.shape) == 3 and lidar_image.shape[2] == 3:
        # RGB图像，使用默认色彩映射
        plt.imsave(f'./sac1_results/lidar_images/lidar_ep{episode}_uav{drone_id}.png', 
                   lidar_image, vmin=0, vmax=1)
    else:
        # 灰度图像或其他类型，使用灰度色彩映射
        plt.imsave(f'./sac1_results/lidar_images/lidar_ep{episode}_uav{drone_id}.png', 
                   lidar_image, cmap='gray', vmin=0, vmax=1)

def savepic(now_ep):
    figure_save_path = "sac1_results"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    # 保存图片时使用bbox_inches='tight'确保图例完整显示
    plt.savefig('./sac1_results/pic-{}.png'.format(now_ep), bbox_inches='tight')


def draw_uav(x,y,u):
    p1 = plt.scatter(x, y, s=8, marker='.', c=u)

def drawobs(x, y, i):
    r = 28.0+i*5
    i =i +1
    if (i < 3):
        plt.scatter(x, y, s=r*2*150*(0.23*i+1.2), marker='.', c='gray')
    if (i > 2):
        plt.scatter(x, y, s=r*2*150*(0.23*i+1.2), marker='.', color=(254/255,232/255,154/255))
    if (i < 3):
        plt.text(x-45, y-1.5*r,'Obstacles',fontsize=15)
    if (i > 2):
        plt.text(x - 85, y - 1.79 * r, 'Interference source', fontsize=15)
    if( i >2):
    # 2.圆心坐标
        a = x
        b = y
        # ==========================================
        # 方法一：参数方程
        n = 35.0+i*5
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a + n * np.cos(theta)
        y = b + n * np.sin(theta)
        # axes = fig.add_subplot(111)
        plt.plot(x, y,linestyle='solid',c = 'orange' , linewidth=2)

def drawnext(next_x ,next_y):
    plt.scatter(next_x, next_y, s=10,marker='*',c = 'red',label='Next position')

def draw_uav_path(positions, color='red'):
    """
    绘制无人机路径
    :param positions: 位置历史列表 [(x1, y1), (x2, y2), ...]
    :param color: 路径颜色
    """
    if len(positions) > 1:
        xs, ys = zip(*positions)
        plt.plot(xs, ys, color=color, linewidth=2)

# 添加绘制拒止区域的函数
def draw_denied_area(center_x, center_y, radius, color='red', alpha=0.2):
    """
    绘制拒止区域
    :param center_x: 拒止区域中心x坐标
    :param center_y: 拒止区域中心y坐标
    :param radius: 拒止区域半径
    :param color: 区域颜色
    :param alpha: 透明度
    """
    # 创建一个圆形表示拒止区域
    circle = plt.Circle((center_x, center_y), radius, color=color, alpha=alpha)
    plt.gca().add_patch(circle)
    
    # 绘制边界圆
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    plt.plot(x, y, linestyle='--', color=color, linewidth=1)

def newcsv(sumway,End,Rew,f,lostend,lostway,losttime):

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(sumway)

    csv_writer.writerow(End)

    csv_writer.writerow(Rew)

    csv_writer.writerow([lostend])

    csv_writer.writerow([lostway])

    csv_writer.writerow([losttime])



def drawUAV(uav,uav_num):
    for u in range (uav_num):
        draw_uav(uav[u][0], uav[u][1],(204/255,103/255,102/255))


# 定义不同无人机的颜色
UAV_COLORS = [
    'red', 'blue', 'green', 'orange', 'purple',
    'brown', 'pink', 'cyan', 'olive', 'gray'
]

def draw_uav_paths_with_colors(uav_positions_history, uav_ids=None):
    """
    用不同颜色绘制多架无人机的路径
    :param uav_positions_history: 无人机位置历史字典 {uav_id: [(x, y), ...]}
    :param uav_ids: 要绘制的无人机ID列表，如果为None则绘制所有
    """
    if uav_ids is None:
        uav_ids = list(uav_positions_history.keys())
    
    lines = []
    labels = []
    
    # 按照uav_id排序，确保颜色和编号的对应关系一致
    uav_ids = sorted(uav_ids)
    
    for i, uav_id in enumerate(uav_ids):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        positions = uav_positions_history.get(uav_id, [])
        # 只有位置点数大于1时才绘制连线
        if len(positions) > 1:
            xs, ys = zip(*positions)
            line, = plt.plot(xs, ys, color=color, linewidth=2)
            lines.append(line)
            # 确保标签显示正确的无人机编号
            label = 'UAV {}'.format(uav_id)
            labels.append(label)
    
    # 在图外右上角显示图例
    if lines and labels:  # 确保有内容才显示图例
        plt.legend(lines, labels, bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)

def draw_all_denied_areas(denied_areas):
    """
    绘制所有拒止区域
    :param denied_areas: 拒止区域信息列表
    """
    for area_info in denied_areas:
        center = area_info['center']
        radius = area_info['radius']
        draw_denied_area(center.x, center.y, radius, color='red', alpha=0.2)


if __name__ == '__main__':
    from main import main
    main()