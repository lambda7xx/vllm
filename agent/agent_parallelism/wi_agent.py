

import time 
import pickle
import argparse
#from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import json 
import time 
from transformers import AutoTokenizer
from multiprocessing import Process, Manager

def load_prompt(bs, pr_len):
    path_300 = "../prompt_300.json"
    path_400 = "../prompt_400.json"
    path_500 ="../prompt_500.json"
    path_1000 = "../prompt_1000.json"
    path_2000 = "../prompt_2000.json"
    if pr_len == 300:
        path = path_300
    if pr_len == 400:
        path = path_400
    if pr_len == 500:
        path = path_500
    elif pr_len == 1000:
        path = path_1000
    elif pr_len == 2000:
        path = path_2000
    new_prompts = [] 
    with open(path, "r") as f:
        prompts = json.load(f)
    # for i in range(1,bs+1):
    #     new_prompts.append(tokenizer.encode(prompts[i], return_tensors="pt"))
    # return new_prompts
    return prompts[1:bs+1]

def warm_up(engine):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=25) #this can be changed
    prompt ="Metaphorical language has been summoned to describe the many enigmatic addressing modes of the instructions at hand. ."
    print(f"len(prompt):{len(prompt)}")
    engine.add_request(request_id = "0112899", inputs = prompt, params = sampling_params, arrival_time = time.time())
    print(f"start warm up")
    while True:
        request_outputs: List[RequestOutput] = engine.step()
        if not engine.has_unfinished_requests():
            break
    print(f"finish warm up")


class ReactReq:
    def __init__(self, rid, arr):
        self.arr = arr
        self.rid = rid 
        self.total_duration = 0
        self.total_token = 0 

def get_prompt(data, marked):
    requests = [None] * len(data)
    i = 0
    for request, prompt in data.items():
        #requests[key] = value
        if marked[request] != prompt:
            requests[i] = [request, prompt]
            marked[request] = prompt
            i += 1
    return requests

def agentb(react_num, promptA, promptB, outputA, outputB, args):
    need_react_num = react_num // 2 + react_num % 2
    model = args.model 
    bs = args.bs 
    i = 0 
    print(f"model:{model}")
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    warm_up(engine)  
    sp2 ="""
    Meta creates you and you are very good at answering the questinons, help me answer the below question.
    """
    print(f"finish warm up")
    markedA = dict()
    markedB = dict()
    requests = get_prompt(promptA, markedA)
    while i != args.bs:
        pass

def agentA(react_num, promptA, promptB, outputA, outputB, requestA, requestB, finishedA, finishedB, reactA, reactB):
    a_need_rect = react_num // 2 + react_num % 2
    prompt_len = args.prompt_len
    max_token = args.max_tokens
    bs = bs
    i = 0 
    model = args.model
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token) #this can be changed
    discard_sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1) #used for agent paralllelism 
    init_prompt = get_prompt(args.bs, prompt_len)
    sp1 = """
    you are very powerful AI, Please help answer the below question.
    """
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    warm_up(engine)  
    info = dict() 
    a_rect = 1
    current_react = 0
    reqs = []
    for i in range(0, bs):
        reqs.append([str(i), init_prompt[i]])
        promptA[str(i)] = init_prompt[i]
        
    
    while i != bs:
        while reqs:
            now = time.time()
            rid, prompt = reqs.pop(0)
            if rid not in info:
                info[rid] = ReactReq(rid, now)
            else:
                info[rid].now = now
            if a_rect % 2 == 1:
                #do prefill + decode 
                engine.add_request(request_id = rid, inputs = prompt, params = sampling_params, arrival_time = now)
        
            #first a do prefill + decode 
            try:
                request_outputs = engine.step()
            except Exception as e:
                print(f"error: {e}")

            for request_output in request_outputs:
                if request_output.finished:
                    #we should know the current react num = agent A  + agent b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark.')

    parser.add_argument('--model', type=str, default='/home/xiaoxiang/data/Llama-3.1-8B', help='model name')
    parser.add_argument('--num-act', type=int, help='number of react steps', default=3)
    parser.add_argument('--batch-size', type=int, help='batch size', default=1)
    parser.add_argument('--agent-parallelism', action='store_true', help='whether use agent parallelism or not')
    parser.add_argument("--agent-tokens", type=int, help="number of tokens for agent parallelism", default=64)
    parser.add_argument("--max-tokens", type=int, help="max tokens for one llm call", default=256)
    parser.add_argument("--prompt-len", type=int, help="prompt length", default=500)
    args = parser.parse_args()
    manager = Manager()
    promptA = manager.dict()    
    promptB = manager.dict()
    outputA = manager.dict()
    outputB = manager.dict()
    requestA= manager.dict()
    requestB= manager.dict()




# #协议：
# """
# 1)A: prefill + decode,  react_num = 1
# send 
# data = {
#     "request_id": "xxx",
#     "prompt": "xxx",
#     "output" = "xxx"
#     “finished”: False,
# }  to B

# 2)B recv data from A, 
# data= recv()
# rid = data["request_id"]
# bs 
# need_react_num  = react_num % 2 + react_num // 2
# num = 0
# prompt = data["prompt"]
# finished = data["finished"]
# output = data["output"]
# sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token) #this can be changed
# discard_sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1) #used for agent paralllelism 
# if "rid" not in react_num:
#     react_num["rid"] = 1 
# if finished == False:
#     do agent prefill for this request use discard_sampling_params 
# else: #do prefill+decode for this request, increase the react num
#     do agent prefill+decode for this request use sampling_params
#     if "rid" not in react_num:
#         react_num["rid"] = 1 
#     else:
#         react_num["rid"] += 1
#     if react_num["rid"] == need_react_num:
#         num = 0 
#     #send the prompt + output to A
#     the data is {
#         "request_id": "xxx",
#         "prompt": "prompt" + output,
#         "finished": True
#     }

# """