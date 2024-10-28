import pandas as pd
import matplotlib.pyplot as plt
import io

# 数据部分
csv_data = '''batch size,wo agent parallelism(512),wo agent parallelism(1024),w agent paralleism(1024)
2,0.05869871378,0.0472615242,0.07000530958
4,0.07648238215,0.0858144767,0.0133494063
8,0.173346769,0.145914807,0.2809921109
16,0.4382217333,0.3649027275,0.3077329293
32,1.031974399,0.8684129228,0.5064376324
64,1.775225582,1.486473063,0.3824802735'''

# 将数据读入 DataFrame
data = pd.read_csv(io.StringIO(csv_data))

# 将 batch size 转换为字符串类型
data['batch size'] = data['batch size'].astype(str)

# 设置图形
plt.figure(figsize=(9, 4))

# 画出柱状图
bar_width = 0.15
index = data.index

plt.bar(index - bar_width, data['wo agent parallelism(512)'], bar_width, label='Without Agent Parallelism (512)')
plt.bar(index, data['wo agent parallelism(1024)'], bar_width, label='Without Agent Parallelism (1024)')
plt.bar(index + bar_width, data['w agent paralleism(1024)'], bar_width, label='With Agent Parallelism (1024)')

# 计算百分比差异并在柱子上方显示
for i in range(3, len(data)):
    value_wo_agent_512 = data.at[i, 'wo agent parallelism(512)']
    value_with_agent = data.at[i, 'w agent paralleism(1024)']
    percentage_diff = (value_wo_agent_512 - value_with_agent) / value_wo_agent_512 * 100
    plt.text(i + bar_width, value_with_agent + 0.02, f'{percentage_diff:.2f}%', ha='center', va='bottom', fontsize=13)

# 设置标题和标签
plt.xlabel('Batch Size',fontsize=14)
plt.ylabel('Agent Normalized Latency (token/s)',fontsize=14)
plt.title('different chunk prefill, prompt=500')
plt.xticks(index, data['batch size'],fontsize=14)
plt.legend()

# 展示图形
plt.tight_layout()
plt.show()
plt.savefig("diff_chunk_prefill_prompt_500.pdf")