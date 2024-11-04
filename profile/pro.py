from vllm import LLM, SamplingParams

import json 

sampling_params = SamplingParams(temperature=0, top_p=0.95,
                                     max_tokens=20)  #this can be changed

import sys
print(sys.path)

model_name = "/home/xiaoxiang/data/Llama-3.1-8B"

# llm = LLM(model=model_name, enable_prefix_caching=True)

llm = LLM(model=model_name, enable_chunked_prefill = True)
# path = "/home/xiaoxiang/vllm/data/prompt_500.json"

path = "/home/xiaoxiang/vllm/data/prompt_1000.json"

path = "/home/xiaoxiang/vllm/data/prompt_1500.json"

# path = "/home/xiaoxiang/vllm/data/prompt_200.json"
#enable_prefix_caching=True
prompts = [] 

prompt1 = ""



with open(path, "r") as f:
    data = json.load(f)
    # prompt = data[0]['value']
    # len = data[0]['len']
    # print(f"prompt length: {len}")

    for d in data:
        prompt1 = d['value']
        len = d['len']
        if len > 2000:
            print(f"1 prompt length: {len}")
            break 

print(f"2 prompt length: {len}")
path = "/home/xiaoxiang/vllm/data/prompt_200.json"

prompt2=""

with open(path, "r") as f:
    data = json.load(f)

    for d in data:
        prompt2 = d['value']
        len = d['len']
        if len > 200:
            print(f"4 prompt length: {len}")
            break 

print(f"5 prompt length: {len}")
    
prompts.append(prompt1)
prompts.append(prompt2)
#warm up 
for i in range(2):
    llm.generate(prompt1, sampling_params)

print(f"*********finished warm up**********")

#start to profile
import time
start = time.time()
print(f"start time: {start}")
llm.generate(prompts, sampling_params)
end = time.time()

print(f"Time elapsed: {end-start}")

