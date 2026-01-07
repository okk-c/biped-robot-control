
'''
运行脚本：python export_rewards.py，可在 Excel 中打开 CSV 文件做进一步分析
'''


# 新建 export_rewards.py，放在项目根目录
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# 替换为你的日志文件路径（从 logs/g1/时间戳文件夹 下找）
log_path = "logs/g1/Nov30_20-52-22_/events.out.tfevents.1733000000.localhost"

# 加载日志
ea = EventAccumulator(log_path)
ea.Reload()

# 提取平均奖励数据
reward_tag = "train/avg_reward"
reward_data = ea.Scalars(reward_tag)

# 转换为DataFrame（方便导出/分析）
df = pd.DataFrame([
    {"step": event.step, "reward": event.value, "time": event.wall_time}
    for event in reward_data
])

# 导出为CSV文件
df.to_csv("g1_train_rewards.csv", index=False)
print("奖励数据已导出到 g1_train_rewards.csv")