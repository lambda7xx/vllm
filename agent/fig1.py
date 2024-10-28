import pandas as pd
import matplotlib.pyplot as plt
import io

# 数据部分
csv_data = '''batch size,wo agent parallelism,w agent paralleism
2,0.01262478238,0.01258125575
4,0.01305089987,0.0133494063
8,0.01387072192,0.01632031671
16,0.01616301817,0.01660708937
32,0.0222131993,0.02904922882
64,0.04335898459,0.04998703181'''

# 将数据读入 DataFrame
data = pd.read_csv(io.StringIO(csv_data))

# 将 batch size 转换为字符串类型
data['batch size'] = data['batch size'].astype(str)

# 设置图形
plt.figure(figsize=(7, 3))

# 画出柱状图
bar_width = 0.15
index = data.index

plt.bar(index - bar_width / 2, data['wo agent parallelism'], bar_width, label='Without Agent Parallelism')
plt.bar(index + bar_width / 2, data['w agent paralleism'], bar_width, label='With Agent Parallelism ')

# 设置标题和标签
plt.xlabel('Batch Size',fontsize=14)
plt.ylabel('Agent Normalized Latency (token/s)',fontsize=14)
plt.title('chunk_prefill=512, prompt=300')
plt.xticks(index, data['batch size'],fontsize=14)
plt.legend()

# 展示图形
plt.tight_layout()
plt.show()

plt.savefig("chunkprefill_512_prompt_300.pdf")