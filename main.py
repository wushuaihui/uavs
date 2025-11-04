# 设置环境变量以避免OpenMP库重复加载的问题
import os

# 设置OpenMP线程数以避免Intel libiomp和LLVM libomp库冲突
# 当同时加载这两个不兼容的OpenMP库时，可能导致程序随机崩溃或死锁
# 将OMP_NUM_THREADS设置为1可以缓解此问题
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 添加更多环境变量设置以进一步减少OpenMP库冲突
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 设置PyTorch使用所有可用的线程以提高性能
os.environ['OMP_NUM_THREADS'] = '8'  # 根据CPU核心数调整
os.environ['MKL_NUM_THREADS'] = '8'  # 根据CPU核心数调整

# 导入命令行参数解析和CSV处理模块
import argparse
import csv

# 导入绘图模块和PyTorch深度学习框架
import draw as d
import torch

# 设置PyTorch使用的线程数，充分利用多核CPU
torch.set_num_threads(8)  # 根据CPU核心数调整

# 从td33模块导入TD3训练器（注释掉的其他导入是替代方案）
#from td3 import TD3Trainer
#from td33 import TD3Trainer
#from two.DDPG import TD3Trainer
from SAC1 import SACTrainer

# 创建命令行参数解析器，用于接收外部输入参数
parser = argparse.ArgumentParser(description='Input n_agents and main folder')
# 添加agents参数，指定无人机数量，默认为9
parser.add_argument('--agents', type=int, default=9)
# 添加folder参数，指定主文件夹路径
parser.add_argument('--folder', type=str)
# 添加global_参数，指定全局设置，默认为"GLOBAL"
parser.add_argument('--global_', type=str, default="GLOBAL")
# 添加checkpoint参数，指定要加载的检查点
parser.add_argument('--checkpoint', type=str, default=None, help='要加载的检查点文件编号')

# 解析命令行参数
args = parser.parse_args()

# 将解析的参数赋值给全局变量
N_AGENTS = args.agents  # 无人机数量
MAIN_FOLDER = args.folder  # 主文件夹路径

def main():
    # 打印启动信息，显示无人机数量和文件夹路径
    print(f'START AGENTS: {N_AGENTS} FOLDER: {MAIN_FOLDER} ')
    
    # 设置PyTorch使用的线程数为8，控制计算资源使用
    torch.set_num_threads(8)
    
    # 创建TD3训练器实例，传入无人机数量和检查点路径作为参数
    # trainer = TD3Trainer(N_AGENTS, load_checkpoint=args.checkpoint)
    # trainer = PPOTrainer(N_AGENTS, load_checkpoint=args.checkpoint)
    trainer = SACTrainer(N_AGENTS, load_checkpoint=args.checkpoint)
    # 打印检查点信息
    if args.checkpoint:
        print(f"从检查点 checkpoint-{args.checkpoint} 加载模型")
        # 从检查点恢复时，应该从检查点的episode开始训练
        start_episode = trainer._now_ep - 1  # 减1是因为train_one_episode会增加计数器
        print(f"从episode {trainer._now_ep} 开始训练")
        # 设置恢复标记，避免计数器重复增加
        trainer._resumed_from_checkpoint = True
    else:
        start_episode = -1  # 设置为-1，这样第一次训练时会变成episode 0
        print("从头开始训练")
        
    # 如果检查点加载失败（文件不存在），则从头开始训练
    if args.checkpoint and trainer._now_ep == 0:
        start_episode = -1
        print("检查点文件不存在，从头开始训练")

    # 打开CSV文件用于记录训练数据，使用gbk编码，newline=""避免空行
    f = open("file.csv", "w", encoding="gbk", newline="")
    
    # 创建无人机标识列表，如['uav_0', 'uav_1', ..., 'uav_N']
    uav= [f'uav_{i}' for i in range(N_AGENTS)]
    
    # 将无人机标识写入CSV文件的第一行作为表头
    csv.writer(f).writerow(uav)
    
    # 进行训练迭代，如果从检查点恢复则从相应episode开始
    # 增加训练次数到20000次，确保有足够的时间训练
    for i in range(start_episode + 1, start_episode + 1 + 20000):
        # 执行一轮训练并获取奖励值
        r = trainer.train_one_episode()
        
        # 减少打印频率，每500个episode打印一次训练信息
        # 计算所有无人机的平均奖励
        avg_reward = sum(r.values()) / len(r) if r else 0
        
        print(f'训练episode: {trainer._now_ep} ')
        # 显示每个无人机的奖励值
        print(f"平均奖励: {avg_reward}")
        print(f"奖励: {r}")

        # 执行测试（静默模式），每10个episode执行一次测试
        if (trainer._now_ep) % 10 == 0:
            # 执行测试（静默模式）
            test_result = trainer.test_one_episode()
            print(f"episode {trainer._now_ep} 测试完成")
            test_avg_reward = sum(test_result.values()) / len(test_result) if test_result else 0
            print(f"测试平均奖励: {test_avg_reward}")
            print(f"测试结果: {test_result}")
            
            # 调用绘图模块的newcsv函数记录训练数据到CSV文件
            # 但仅在必要时记录，避免频繁I/O
            d.newcsv(trainer.sumway,trainer.End,trainer.rew,f,trainer.lostend,trainer.lostway,trainer.losttime)

        # 保存检查点，每1000个episode保存一次
        if trainer._now_ep % 1000 == 0:
            trainer.save_checkpoint(trainer._now_ep)
            print(f'检查点已在episode {trainer._now_ep} 保存')

    # 关闭CSV文件
    f.close()

# 当脚本作为主程序运行时执行main函数
if __name__ == '__main__':
    main()