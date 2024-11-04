from vllm import LLM, SamplingParams

import json 

sampling_params = SamplingParams(temperature=0, top_p=0.95,
                                     max_tokens=20)  #this can be changed

import sys
print(sys.path)

model_name = "/home/xiaoxiang/data/Llama-3.1-8B"


llm = LLM(model=model_name, enable_chunked_prefill = True)



path1 = "/home/xiaoxiang/vllm/data/prompt_300.json"

path2 = "/home/xiaoxiang/vllm/data/prompt_400.json"

path3 = "/home/xiaoxiang/vllm/data/prompt_500.json"

prompts = [] 

warm_prompt =""

with open(path2, "r") as f:
    data = json.load(f)
    for d in data:
        prompts.append(d["value"])
        warm_prompt = d["value"]
        prompt_le = d["len"]
        print(f"len:{prompt_le}")
        if len(prompts) == 32:
            break 
    
#warm up 
for i in range(4):
    llm.generate(warm_prompt, sampling_params)

print(f"*********finished warm up**********")

#start to profile
import time
start = time.time()
print(f"start time: {start}")
llm.generate(prompts, sampling_params)
end = time.time()

print(f"Time elapsed: {end-start}")
