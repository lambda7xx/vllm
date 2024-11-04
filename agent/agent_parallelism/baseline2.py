import time 
import pickle
import argparse
from vllm import EngineArgs, LLMEngine, SamplingParams
import json 
import multiprocessing as mp
import zmq
import os 
from dataclasses import dataclass


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


def load_prompt(bs, pr_len):
    path_mapping = {
        300: "../prompt_300.json",
        400: "../prompt_400.json",
        500: "../prompt_500.json",
        1000: "../prompt_1000.json",
        2000: "../prompt_2000.json"
    }
    path = path_mapping.get(pr_len)
    with open(path, "r") as f:
        prompts = json.load(f)
    return prompts[1:bs+1]

def warm_up(engine):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=25)
    prompt ="Metaphorical language has been summoned to describe the many enigmatic addressing modes of the instructions at hand."
    engine.add_request(request_id="0112899", prompt=prompt, params=sampling_params, arrival_time=time.time())
    while True:
        engine.step()
        if not engine.has_unfinished_requests():
            break

@dataclass 
class CommArgs:
    host: str = "127.0.0.1"
    recv_from_scheduler_port: int = 30000
    send_to_scheduler_port: int = 30001

class BSendData:
    def __init__(self, request_id, prompt, output_text_len, output_text):
        self.request_id = request_id
        self.prompt = prompt
        self.output_text_len = output_text_len
        self.output_text = output_text

def agentb(react_num, reactB, comm_args, args):
    b_need_react_num = react_num // 2 
    os.environ["CUDA_VISIBLE_DEVICES"]  = "1"
    model = args.model 
    max_token = args.max_tokens
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token)

    context = zmq.Context(2)
    recv_from_a = context.socket(zmq.PULL)
    recv_from_a.bind(f"tcp://{comm_args.host}:{comm_args.recv_from_scheduler_port}")
    send_to_a = context.socket(zmq.PUSH)
    send_to_a.connect(f"tcp://{comm_args.host}:{comm_args.send_to_scheduler_port}")
    warm_up(engine)

    print(f"agentB start")
    need_reqs = 0 
    reqs = [] 
    while need_reqs != args.batch_size:
        try:
            raw_message = recv_from_a.recv(zmq.NOBLOCK)
            recv_reqs = pickle.loads(raw_message)
            print(f"agentB recv data from A, data length: {len(recv_reqs)}")
            for rid, prompt in recv_reqs.items():
                print(f"agentB received rid: {rid}")
                reqs.append((rid, prompt))
        except zmq.Again:
            iiii = 0

        while reqs:
            now = time.time()
            rid, prompt = reqs.pop(0)
            if rid not in reactB:
                reactB[rid] = 1
                engine.add_request(request_id=rid, prompt=prompt, params=sampling_params, arrival_time=now)
                print(f"agentB added rid: {rid} to engine")
            elif reactB[rid] < b_need_react_num:
                engine.add_request(request_id=rid, prompt=prompt, params=sampling_params, arrival_time=now)
                print(f"agentB re-added rid: {rid} to engine")
                reactB[rid] += 1
        
        request_outputs = []
        try:
            if engine.has_unfinished_requests():
                request_outputs = engine.step()
        except Exception as e:
            print(f"agentB error: {e}")

        send_data = {}
        for request_output in request_outputs:
            if request_output.finished:
                rid = request_output.request_id
                output_text_len = len(request_output.outputs[0].token_ids)
                output_text = request_output.outputs[0].text
                if reactB[rid] == b_need_react_num:
                    need_reqs += 1
                    print(f"agentB finished rid: {rid}")
                b_send_data = BSendData(rid, prompt, output_text_len, output_text)
                send_data[rid] = b_send_data

        if send_data:
            try:
                send_to_a.send(pickle.dumps(send_data))
                print(f"agentB sent data to A")
            except Exception as e:
                print(f"agentB error sending data: {e}")

def agentA(react_num, reactA, comm_args, args):
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
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token)
    init_prompt = load_prompt(bs, prompt_len)
    sp1 = "You are a powerful AI. Please help answer the below question."
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    warm_up(engine)  
    print(f"agentA warm up finished")
    
    info = {}
    reqs = [(str(i), init_prompt[i]) for i in range(bs)]
    for rid, _ in reqs:
        reactA[rid] = 0
    
    num_reqs = 0 
    start_i = 0 
    while num_reqs != bs:
        if start_i == 0:
            start_i = 1
        else:
            try:
                raw_message = recv_from_b.recv(zmq.NOBLOCK)
                recv_reqs = pickle.loads(raw_message)
                reqs =[None] * len(recv_reqs)
                i = 0
                now = time.time()
                for req in recv_reqs.values():
                    rid = req.request_id
                    output_text = req.output_text
                    prompt = req.prompt + output_text
                    output_text_len = req.output_text_len
                    if rid not in info:
                        info[rid] = ReactReq(rid, now)
                    info[rid].total_duration += now - info[rid].send_time
                    info[rid].total_token += output_text_len
                    info[rid].send_time = None
                    reqs[i] = (rid, prompt)
                    i += 1
            except zmq.Again:
                iii = 0 

        while reqs:
            now = time.time()
            rid, prompt = reqs.pop(0)
            if reactA[rid] < a_need_rect:
                engine.add_request(request_id=rid, prompt=sp1 + prompt, params=sampling_params, arrival_time=now)
                reactA[rid] += 1

        request_outputs = []
        try:
            request_outputs = engine.step()
        except Exception as e:
            print(f"agentA error: {e}")

        send_data = {}
        for request_output in request_outputs:
            if request_output.finished:
                rid = request_output.request_id
                now = time.time()
                output_text_len = len(request_output.outputs[0].token_ids)
                output_text = request_output.outputs[0].text
                if reactA[rid] == a_need_rect:
                    num_reqs += 1
                    print(f"agentA finished rid: {rid}")
                else:
                    info[rid].send_time = now
                    send_data[rid] = prompt + output_text

        if send_data:
            try:
                send_to_b.send(pickle.dumps(send_data))
                print(f"agentA sent data to B")
            except Exception as e:
                print(f"agentA error sending data: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark.')
    parser.add_argument('--model', type=str, default='/home/xiaoxiang/data/Llama-3.1-8B', help='model name')
    parser.add_argument('--num-act', type=int, help='number of react steps', default=3)
    parser.add_argument('--batch-size', type=int, help='batch size', default=1)
    parser.add_argument('--max-tokens', type=int, help="max tokens for one llm call", default=256)
    parser.add_argument('--prompt-len', type=int, help="prompt length", default=500)
    args = parser.parse_args()
    
    manager = mp.Manager()
    reactA = manager.dict()
    reactB = manager.dict()
    comm_args = CommArgs()

    agent_b_process = mp.Process(target=agentb, args=(args.num_act, reactB, comm_args, args))
    agent_b_process.start()
    
    agentA(args.num_act, reactA, comm_args, args)
