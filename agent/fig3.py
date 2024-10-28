import pandas as pd
import matplotlib.pyplot as plt
import io

# 数据部分
csv_data = '''batch size,wo agent paralleism(512),wo agent paralleism(1024),wo agent paralleism(1536),w agent paralleism(1536)
2,0.07490904427,0.07163715161,0.05759088974,0.08291303165
4,0.2088831164,0.1736352364,0.1646778878,0.294478617
8,0.2646250147,0.2119064614,0.1920043033,0.0903975629
16,0.4505608328,0.3495500702,0.3204031222,0.1229075386
32,0.9586816407,0.7793498923,0.7084031924,0.3855887193
64,1.776482819,1.394300474,1.277224888,0.6210578736'''

# 将数据读入 DataFrame
data = pd.read_csv(io.StringIO(csv_data))

# 将 batch size 转换为字符串类型
data['batch size'] = data['batch size'].astype(str)

# 设置图形
plt.figure(figsize=(13, 7))

# 画出柱状图
bar_width = 0.15
index = data.index

plt.bar(index - 1.5 * bar_width, data['wo agent paralleism(512)'], bar_width, label='Without Agent (512)')
plt.bar(index - 0.5 * bar_width, data['wo agent paralleism(1024)'], bar_width, label='Without Agent (1024)')
plt.bar(index + 0.5 * bar_width, data['wo agent paralleism(1536)'], bar_width, label='Without Agent (1536)')
plt.bar(index + 1.5 * bar_width, data['w agent paralleism(1536)'], bar_width, label='With Agent (1536)')

# 计算百分比差异并在柱子上方显示
for i in range(2, len(data)):
    value_wo_agent_512 = data.at[i, 'wo agent paralleism(512)']
    value_with_agent = data.at[i, 'w agent paralleism(1536)']
    percentage_diff = (value_wo_agent_512 - value_with_agent) / value_wo_agent_512 * 100
    plt.text(i + 1.5 * bar_width, value_with_agent + 0.02, f'{percentage_diff:.2f}%', ha='center', va='bottom', fontsize=16)

# 设置标题和标签
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('Agent Normalized Latency (token/s)',fontsize=14)
plt.title('diff chunk prefill, prompt=1000')
plt.xticks(index, data['batch size'], fontsize=14)
plt.legend()

# 展示图形
plt.tight_layout()
plt.show()
plt.savefig('diff_chunk_prefill_prompt1000.pdf')