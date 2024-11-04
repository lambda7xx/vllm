

import time 
import pickle
import argparse
#from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import json 
import time 
from multiprocessing import Process, Manager
import zmq
import os 
import multiprocessing as mp
from dataclasses import dataclass

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
    with open(path, "r") as f:
        prompts = json.load(f)
    # for i in range(1,bs+1):
    #     new_prompts.append(tokenizer.encode(prompts[i], return_tensors="pt"))
    # return new_prompts
    return prompts[1:bs+1]

def warm_up(engine):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=25) #this can be changed
    prompt ="Metaphorical language has been summoned to describe the many enigmatic addressing modes of the instructions at hand. ."
    engine.add_request(request_id = "0112899", prompt = prompt, params = sampling_params, arrival_time = time.time())
    print(f"start warm up")
    while True:
        engine.step()
        if not engine.has_unfinished_requests():
            break
    print(f"finish warm up")


class ReactReq:
    def __init__(self, rid, arr):
        self.arr = arr
        self.rid = rid 
        self.total_duration = 0
        self.total_token = 0 
        self.send_time = 0  
        self.finished = False


class BSendData:
    def  __init__(self, request_id, prompt, output_text_len, output_text):
        self.request_id = request_id
        self.prompt = prompt
        self.output_text_len = output_text_len
        self.output_text = output_text


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

@dataclass 
class CommArgs:
    """ Arguments for BlockManager """
    #port
    host: str = "127.0.0.1"
    recv_from_scheduler_port: int = 30000
    send_to_scheduler_port: int = 30001

#currently we only consider the react_num is even number
def agentb(react_num, reactB, comm_args, args):
    b_need_react_num = react_num // 2 
    os.environ["CUDA_VISIBLE_DEVICES"]  = "1"
    model = args.model 
    bs = args.batch_size
    i = 0 
    print(f"model:{model}")
    max_token = args.max_tokens
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token) #this can be changed

    context = zmq.Context(2)
    recv_from_a = context.socket(zmq.PULL)
    recv_from_a.bind(f"tcp://{comm_args.host}:{comm_args.recv_from_scheduler_port}")
    send_to_a = context.socket(zmq.PUSH)
    send_to_a.connect(f"tcp://{comm_args.host}:{comm_args.send_to_scheduler_port}")
    warm_up(engine)  


    sp2 ="""
    Meta creates you and you are very good at answering the questinons, help me answer the below question.
    """
    print(f"agentB start")
    need_reqs = 0 
    reqs = [] 
    while  need_reqs  != bs:
        try:
            raw_message = recv_from_a.recv(zmq.NOBLOCK)
            recv_reqs = pickle.loads(raw_message)
            print(f"0 agentB recv data from A, data:{len(recv_reqs)}")
            reqs = [None] * len(recv_reqs)
            i = 0
            for rid, prompt in recv_reqs.items():
                print(f"1 agentB recv data from A the rid:{rid}")
                reqs[i] = [rid, prompt]
                i += 1
        except zmq.Again:
            iii = 0 
            # print(f"1.2 agentB recv no data from A")

        while reqs:
            now = time.time()
            rid, prompt = reqs.pop()
            #do prefill + decode
            if rid not in reactB:
                reactB[rid] = 1
                engine.add_request(request_id = rid, prompt = sp2 + prompt, params = sampling_params, arrival_time = now)
                print(f"1 agentB rid:{rid} is added to engine")
            elif reactB[rid] < b_need_react_num:
                engine.add_request(request_id = rid, prompt = sp2 + prompt, params = sampling_params, arrival_time = now)
                print(f"1.5 agentB rid:{rid} is added to engine")
                reactB[rid] += 1
        request_outputs = [] 
        try:
            if engine.has_unfinished_requests():
                request_outputs = engine.step()
        except Exception as e:
            print(f"error: {e}")
        send_data = dict()
        for request_output in request_outputs:
            if request_output.finished:
                rid = request_output.request_id
                now = time.time()
                output_text_len = len(request_output.outputs[0].token_ids)
                output_text = request_output.outputs[0].text

                if reactB[rid] == b_need_react_num:
                    need_reqs  += 1 #one application is finished
                    print(f"2 agentB rid:{rid} is finished")
                #send the data to A
                b_send_data = BSendData(rid, prompt, output_text_len, output_text)
                #TODO send the data to A
                print(f"2agentB will send data to A rid:{rid}")
                send_data[rid] = b_send_data
        
        #use zmq send the data to A
        if len(send_data) != 0:
            
            try:
                send_to_a.send(pickle.dumps(send_data))
                print(f"3 agentB send data to A")
            except Exception as e:
                print(f"Error sending data to B: {e}")


def agentA(react_num, reactA, comm_args, args):
    #init zmq 
    os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
    context = zmq.Context(2)
    recv_from_b = context.socket(zmq.PULL)
    recv_from_b.bind(f"tcp://{comm_args.host}:{comm_args.send_to_scheduler_port}")
    send_to_b = context.socket(zmq.PUSH)
    send_to_b.connect(f"tcp://{comm_args.host}:{comm_args.recv_from_scheduler_port}")
                
    a_need_rect = react_num // 2 + react_num % 2
    prompt_len = args.prompt_len
    max_token = args.max_tokens
    bs = args.batch_size
    
    model = args.model
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token) #this can be changed
    # discard_sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1) #used for agent paralllelism 
    init_prompt = load_prompt(bs, prompt_len)
    sp1 = """
    you are very powerful AI, Please help answer the below question.
    """
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    print(f"start warm up")
    warm_up(engine)  
    print(f"warm up finished")
    info = dict() 
    reqs = []
    # def set_reqs():
        
    for i in range(0, bs):
        reqs.append([str(i), init_prompt[i]])
        reactA[str(i)] = 0
    start = 0
    num_reqs = 0 
    while num_reqs != bs:
        if start == 0:#this is the start point:
            start = 1
        else:
            try:
                raw_message = recv_from_b.recv(zmq.NOBLOCK)
                recv_reqs = pickle.loads(raw_message)
                now = time.time()
                #the data format should be BSendData list
                #TODO update the total_duration,total token and and send_time bc it means the agent B has finished the request
                # for rid, prompt, output_text_len, output_text in reqs:
                reqs = [None] * len(recv_reqs)
                i = 0
                for req in recv_reqs.values():
                    rid = req.request_id
                    print(f"1 agentA recv data from B rid:{rid}")
                    output_text = req.output_text
                    prompt = req.prompt + output_text
                    output_text_len = req.output_text_len
                    info[rid].total_duration += now - info[rid].send_time
                    info[rid].total_token += output_text_len
                    info[rid].send_time = None
                    reqs[i] = [rid, prompt]
                    i += 1
            except zmq.Again:
                # print(f"recv no data from b")
                iiii = 0

        while reqs:
            now = time.time()
            rid, prompt = reqs.pop()
            if rid not in info:
                info[rid] = ReactReq(rid, now)
            else:
                info[rid].now = now
            #do prefill + decode 
            if reactA[rid] < a_need_rect:
                engine.add_request(request_id = rid, prompt = sp1 + prompt, params = sampling_params, arrival_time = now)
                reactA[rid] += 1
        
            #first a do prefill + decode 
        try:
            request_outputs = engine.step()
        except Exception as e:
            print(f"error: {e}")
        send_data = dict()
        for request_output in request_outputs:
            if request_output.finished:
                rid = request_output.request_id
                print(f"2 agentA rid:{rid} is finished and reactA[rid]:{reactA[rid]} and a_need_rect:{a_need_rect}")
                now = time.time()
                output_text_len = len(request_output.outputs[0].token_ids)
                info[rid].total_duration += now - info[rid].arr
                info[rid].total_token +=output_text_len
                output_text = request_output.outputs[0].text
                if reactA[rid] == a_need_rect:
                    num_reqs += 1 #one application is finished
                    info[rid].finished = True
                    print(f"3 agentA application rid:{rid} is finished")
                else:
                    print(f"4 agentA will send rid:{rid} to B and reactA[rid]:{reactA[rid]} and a_need_rect:{a_need_rect}")
                    info[rid].send_time = now
                    send_data[rid] = prompt + output_text

        #use zmq send the data to B
        if len(send_data) != 0:
            try:
            
                # print(f"send_data:{send_data}")
                send_to_b.send(pickle.dumps(send_data))
                print(f"5 agentA send data to B and len(send_data):{len(send_data)}")
            except Exception as e:
                print(f"Error sending data to B: {e}")

    counter =0
    for _, value in info.items():
        if value["finished"] == True:
            counter += 1
    assert counter == bs

    #finished all the application  and statistics the time 
    normalized_latencies = []
    latencies = []
    
    print(f"len(info):{len(info)} and bs:{bs}")
    for _, value in info.items():
        normalized_latency = value.total_duration / value.total_token
        normalized_latencies.append(normalized_latency)
        latencies.append(value.total_duration)
    print(f"baseline bs:{bs} and avg_normalized_latency:{sum(normalized_latencies) / len(normalized_latencies)} and avg_latency:{sum(latencies) / len(latencies)}")
        #print(f"request_id:{key}, total_duration:{value.total_duration}, total_token:{value.total_token}")      


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
    # promptA = manager.dict()    
    # promptB = manager.dict()
    # outputA = manager.dict()
    # outputB = manager.dict()
    # requestA= manager.dict()
    # requestB= manager.dict()
    reactA = manager.dict()
    reactB = manager.dict()
    comm_args = CommArgs()
    agent_b_process = mp.Process(target=agentb, args=(args.num_act, reactA, comm_args, args))
    agent_b_process.start()
    # agentb(args.num_act, reactB, comm_args, args)
    agentA(args.num_act,reactA, comm_args, args)




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