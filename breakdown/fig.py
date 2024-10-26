
import pickle 
import os 
import matplotlib.pyplot as plt
import numpy as np

num_acts = [3,4,5,6]

request_rates = [1, 2, 3, 4, 5, 6]

model_name = "llama3_1_8B"

# path = f"./data/vllm_react_model_name_{model_name}_request_rate{rw}_num_act{int(num_act)}.pkl"

inference_latency_dict = dict() 

queue_latency_dict = dict() 

latency_percentage_dict = dict()

for num_act in num_acts:
    inference_latency_dict[num_act] = []
    queue_latency_dict[num_act] = []
    latency_percentage_dict[num_act] = []
    for rw in request_rates:
        path = f"./data/vllm_react_model_name_{model_name}_request_rate{rw}_num_act{int(num_act)}.pkl"
        with open(path, "rb") as f:
            collect_data = pickle.load(f)
            total_duration = 0 
            total_queue_duration = 0
            per_token_latenies = []
            expected_token_latenies = []
            per_token_waiting_latenies = []
            total_generated_tokens = 0
            for data in collect_data:
                per_token_latenies.append(data["per_token_latency"])
                expected_token_latenies.append(data["duration"] / data["generated_text_len"])
                per_token_waiting_latenies.append(data["queue_duration"] / data["generated_text_len"])
            average_latency_1 = sum(per_token_latenies[:1000]) / 1000 #len(per_token_latenies)
            average_latency_2 = sum(expected_token_latenies[:1000]) / 1000  #len(expected_token_latenies)
            average_wait = sum(per_token_waiting_latenies[:1000]) / 1000  #len(per_token_waiting_latenies)
            # average_latency_1 = sum(per_token_latenies) / len(per_token_latenies)
            # average_latency_2 = sum(expected_token_latenies) / len(expected_token_latenies)
            # average_wait = sum(per_token_waiting_latenies) / len(per_token_waiting_latenies)
            inference_latency = average_latency_1 - average_wait
            print(f"1 act:{num_act} rw:{rw} average latency:{average_latency_1} and average_latency_2:{average_latency_2 } average wait:{average_wait}")
            inference_latency_dict[num_act].append((average_latency_1 - average_wait)*1000)
            queue_latency_dict[num_act].append(average_wait * 1000)
            latency_percentage_dict[num_act].append(inference_latency / average_latency_1 * 100)

bar_width = 0.35
for num_act in num_acts:
    index = np.arange(len(request_rates))
    fig, ax = plt.subplots()
    bars1 = ax.bar(index, queue_latency_dict[num_act], bar_width, label='queue latency', color='blue')
    bars2 = ax.bar(index, inference_latency_dict[num_act], bar_width, bottom=queue_latency_dict[num_act], label='inference latency', color='red')
    for i, (latency_percentage) in enumerate(latency_percentage_dict[num_act]):
        ax.text(i, bars1[i].get_height() + bars2[i].get_height() + 0.1, f'{latency_percentage:.1f}%', ha='center')
    ax.set_ylabel('Agent Normalized Latency(ms)',fontsize=14)
    ax.set_title(f'React={num_act} Latency Breakdown', fontsize=14)
    ax.set_xlabel('Agent Request Rate/s',fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(request_rates)
    ax.legend()
    plt.savefig(f'./figs/breakdown_{model_name}_num_act{num_act}.pdf')