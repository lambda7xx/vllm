import numpy as np
import random
from typing import List, Tuple
import tiktoken
# from vllm import LLM, SamplingParams
# from vllm.inputs import  TokensPrompt
from transformers import AutoTokenizer

import json 

path = "/home/xiaoxias/vllm_profile/ShareGPT_V3_unfiltered_cleaned_split.json"

tokenizer = AutoTokenizer.from_pretrained("/data/llama3/Meta-Llama-3-8B-Instruct-hf")

# path2 = 

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model) #maybe need to change the model
        tokens = encoding.encode(text)
        return len(tokens)
    except ValueError as e:
        print(f"Skipping text due to ValueError: {e}")
        return None

def load():
    user_prompts = [] 
    with open(path, 'r') as f:
        data = json.load(f)
    for item in data: 
        conversations = item.get('conversations', [])
        for conversation in conversations:
            value = conversation.get('value')
            user_prompts.append(value)
    return user_prompts



def generate_vllm_system_prompt(num_system_prompts: int, model: str = "gpt-3.5-turbo"):
    system_prompts = []
    
    for i in range(num_system_prompts):
        base_string = f"prompt_{i}_ "
        num_token = count_tokens(base_string, model)
        
        repeat_count = 1000 * 3 // num_token #1000 * 2 can changed 
        remainder = 1000 % num_token
        print(f"num_token: {num_token} and repeat_count: {repeat_count} and remainder: {remainder}")
        prompt = base_string * repeat_count + base_string[:remainder]
        
        generated_token_ids = tokenizer.encode(prompt)
        token_count = len(generated_token_ids)
        print(f"Generated prompt with {token_count} tokens")
        if token_count >= 1000:
            prompt =  tokenizer.decode(generated_token_ids[:1000], skip_special_tokens=True)
            system_prompts.append(prompt)
        else:
            print(f"Generated prompt with {token_count} tokens, retrying...")
        print(f"finally the prompt len:{len(tokenizer.encode(prompt))}")
    
    return system_prompts

# def generate_vllm_text_requests(
#         request_rate: int,
#         duration:int, #10
#         num_system_prompt:int=20,
#         time_quantum: int = 100,
#         max_seq_len: int =2048):
#     lam =request_rate * (time_quantum / 1000)
#     print(f"1 generate_vllm_text_requests lam:{lam} and request_rate:{request_rate} and time_quantum:{time_quantum}")
#     quantums_per_sec = 1000 / time_quantum
#     print(f"2 generate_vllm_text_requests quantums_per_sec:{quantums_per_sec} and duration * quantums_per_sec:{int(duration * quantums_per_sec)}")
#     arrival_times =  np.random.poisson(lam=lam, size=int(duration * quantums_per_sec))
#     print(f"3 generate_vllm_text_requests arrival_times.shape:{arrival_times.shape}")
#     timestamps = []
#     for i, n in enumerate(arrival_times):
#         timestamps += [i * (time_quantum / 1000)] * n
    
#     num_requests = len(timestamps) 
#     print(f"len(timestamps): {num_requests}")
#     #load data 
#     prompts = load()
#     prompts = prompts[:num_requests]
#     random.shuffle(prompts)
#     system_prompts = generate_vllm_system_prompt(num_system_prompt)
#     # random_sampling_params_dict = {'temperature': 0}
#     requests = [] 
#     assert len(timestamps) == len(prompts)
#     i = 0 
#     for timestamp, prompt in zip(timestamps, prompts):
#         system_prompt = system_prompts[i % num_system_prompt]
#         whole_prompt = system_prompt + prompt
#         #whole_prompt = prompt
#         requests.append((timestamp,whole_prompt)) #TODO: this can change to align with the vllm document 
#         i = i + 1 
#     random.shuffle(requests)
#     print(f"generate_text_requests len(requests): {len(requests)}")
#     return requests


def generate_vllm_text_requests_fixed(
        request_rate: int,
        duration:int, #10
        # num_system_prompt:int=20,
        time_quantum: int = 100,
        max_seq_len: int =2048):
    # lam =request_rate * (time_quantum / 1000)
    # print(f"1 generate_vllm_text_requests lam:{lam} and request_rate:{request_rate} and time_quantum:{time_quantum}")
    # quantums_per_sec = 1000 / time_quantum
    # print(f"2 generate_vllm_text_requests quantums_per_sec:{quantums_per_sec} and duration * quantums_per_sec:{int(duration * quantums_per_sec)}")
    # arrival_times =  np.random.poisson(lam=lam, size=int(duration * quantums_per_sec))
    # print(f"3 generate_vllm_text_requests arrival_times.shape:{arrival_times.shape}")
    # timestamps = []
    # for i, n in enumerate(arrival_times):
    #     timestamps += [i * (time_quantum / 1000)] * n
    timestamps = []

    # if request_rate <= 5:
    #     duration = 1000

    # if request_rate <=10:
    #     duration = 500
    # elif request_rate > 10:
    #     duration = 400
    if request_rate <= 8:
        duration = int(100 / request_rate) #
    else:
        duration = int(120 / request_rate)

    
    for sec in range(duration):
        for _ in range(int(request_rate)):
            timestamps.append(sec + np.random.rand())
    
    num_requests = len(timestamps) 
    print(f"len(timestamps): {num_requests}")
    #load data 
    prompts = load()
    prompts = prompts[:num_requests]
    # random.shuffle(prompts)
    # system_prompts = generate_vllm_system_prompt(num_system_prompt)
    requests = len(timestamps) * [None] 
    assert len(timestamps) == len(prompts)
    i = 0 
    for timestamp, prompt in zip(timestamps, prompts):
        whole_prompt = prompt 
        # print(f"i:{i} and len(whole_prompt):{len(tokenizer.encode(whole_prompt))} and duration*request_rate:{duration*request_rate}")
        #requests.append((str(i),timestamp,whole_prompt)) #TODO: this can change to align with the vllm document 
        requests[i] = [str(i), timestamp,whole_prompt]#TODO: this can change to align with the vllm document 
        i = i + 1 
    print(f"generate_text_requests len(requests): {len(requests)}")
    return requests

def generate_agent_system_prompt(num_system_prompts: int, model: str = "gpt-3.5-turbo"):
    system_prompts = []
    
    for i in range(num_system_prompts):
        base_string = f"prompt_{i}_ "
        num_token = count_tokens(base_string, model)
        
        repeat_count = 1000 * 3 // num_token #1000 * 2 can changed 
        remainder = 1000 % num_token
        print(f"num_token: {num_token} and repeat_count: {repeat_count} and remainder: {remainder}")
        prompt = base_string * repeat_count + base_string[:remainder]
        
        generated_token_ids = tokenizer.encode(prompt)
        token_count = len(generated_token_ids)
        print(f"Generated prompt with {token_count} tokens")
        if token_count >= 500:
            prompt =  tokenizer.decode(generated_token_ids[:500], skip_special_tokens=True)
            system_prompts.append(prompt)
        else:
            print(f"Generated prompt with {token_count} tokens, retrying...")
        print(f"finally the prompt len:{len(tokenizer.encode(prompt))}")
    return system_prompts

# def generate_agent_text_requests(
#         request_rate: int,
#         duration:int, #10
#         num_system_prompt:int=20,
#         time_quantum: int = 5,
#         max_seq_len: int =2048):
#     lam = request_rate * (time_quantum / 1000)
#     quantums_per_sec = 1000 / time_quantum
#     arrival_times = np.random.poisson(
#         lam=lam, size=int(duration * quantums_per_sec))
#     timestamps = []
#     for i, n in enumerate(arrival_times):
#         timestamps += [i * (time_quantum / 1000)] * n
    
#     num_requests = len(timestamps) 
#     print(f"len(timestamps): {num_requests}")
#     #load data 
#     prompts = load()
#     prompts = prompts[:num_requests]
#     random.shuffle(prompts)
#     system_prompts = generate_agent_system_prompt(num_system_prompt)
#     # random_sampling_params_dict = {'temperature': 0}
#     requests = [] 
#     assert len(timestamps) == len(prompts)
#     i = 0 
#     for timestamp, prompt in zip(timestamps, prompts):
#         system_prompt = system_prompts[i % num_system_prompt]
#         whole_prompt = system_prompt + prompt
#         requests.append((timestamp,whole_prompt)) #TODO: this can change to align with the vllm document 
#         i = i + 1 
#     # random.shuffle(requests)
#     print(f"generate_text_requests len(requests): {len(requests)}")
#     return requests
    


def generate_agent_text_requests_fixed(
        request_rate: int,
        duration:int, #10
        # num_system_prompt:int=20,
        time_quantum: int = 5,
        max_seq_len: int =2048):
    # lam = request_rate * (time_quantum / 1000)
    # quantums_per_sec = 1000 / time_quantum
    # arrival_times = np.random.poisson(
    #     lam=lam, size=int(duration * quantums_per_sec))
    # timestamps = []
    # for i, n in enumerate(arrival_times):
    #     timestamps += [i * (time_quantum / 1000)] * n
    
    timestamps = []

    if request_rate < 5:
        duration = 1000 

    if request_rate >= 5:
        duration = 800
    if request_rate <= 8:
        duration = int(2000 / request_rate) #
    else:
        duration = int(3000 / request_rate) #
    # elif request_rate > 30:
    #     duration = 100
    
    for sec in range(duration):
        for _ in range(int(request_rate)):
            timestamps.append(sec + np.random.rand())

    num_requests = len(timestamps) 
    print(f"len(timestamps): {num_requests}")
    #load data 
    prompts = load()
    prompts = prompts[:num_requests]
    # system_prompts = generate_agent_system_prompt(num_system_prompt)
    requests = len(timestamps) * [None]
    assert len(timestamps) == len(prompts)
    i = 0 
    for timestamp, prompt in zip(timestamps, prompts):
        # system_prompt = system_prompts[i % num_system_prompt]
        # whole_prompt = system_prompt + prompt
        whole_prompt = prompt
        print(f"i:{i} and len(prompt):{len(tokenizer.encode(prompt))} and len(system_prompt):{len(tokenizer.encode(system_prompt))} and len(whole_prompt):{len(tokenizer.encode(whole_prompt))}")
        requests[i] = [str(i), timestamp,whole_prompt, 0]#TODO: this can change to align with the vllm document 
        i = i + 1 
    print(f"generate_text_requests len(requests): {len(requests)}")
    return requests
    