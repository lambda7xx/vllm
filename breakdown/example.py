import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [1, 2, 3, 4, 5]
y = [3, 4, 6, 7, 8]

# 定义柱状图的宽度
bar_width = 0.35

# 定义柱状图的x轴
index = np.arange(len(x))

# 创建柱状图
fig, ax = plt.subplots()

# 画两个数据的柱状图
bars1 = ax.bar(index, x, bar_width, label='X数据', color='blue')
bars2 = ax.bar(index, y, bar_width, bottom=x, label='Y数据', color='orange')

# 添加标签和标题
ax.set_xlabel('柱子编号')
ax.set_ylabel('数据值')
ax.set_title('包含两个数据的柱状图')
ax.set_xticks(index)
ax.set_xticklabels(['柱1', '柱2', '柱3', '柱4', '柱5'])
ax.legend()

# 显示图形
plt.show()
plt.savefig('bar_chart.png')