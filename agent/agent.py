from vllm import LLM, SamplingParams
import time 
import pickle
import argparse
#from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import json 
import time 

class ReactReq:
    def __init__(self, arr1=None, arr2=None, prompt_len=0, output_len=0,rid1= None, rid2=None, user_prompt = None):
        self.arr1 = arr1
        self.arr2 = arr2
        self.prompt_len = prompt_len
        self.output_len = output_len 
        self.rid1 = rid1
        self.rid2 = rid2 #rid1 is the first request id, rid2 is the second request id
        self.total_duration = 0
        self.total_token = 0 
        self.r1_user_prompt = user_prompt
        self.r2_user_prompt = None

def warm_up(engine):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=25) #this can be changed

    
    prompt ="Metaphorical language has been summoned to describe the many enigmatic addressing modes of the instructions at hand. The speakers have resorted to grandiose expressions to convey their immense wonder and admiration for the sheer functionality of the aforementioned directives. Among the mystifying commands are the enigmatic JMP ABCD, the confounding MOV AX, [BX+SI], the inscrutable MOV AX, [100], the cryptic MOV AX, [BX], the perplexing MOV AX, [BX\*2+SI], the unfathomable MOV AX, BX, and finally, the enigmatic MOV AX, 7."
    for i in range(1):
        engine.add_request(request_id = i, inputs = prompt, params = sampling_params, arrival_time = time.time())
    while True:
        request_outputs: List[RequestOutput] = engine.step()
        if not engine.has_unfinished_requests():
            break   
    
def get_prompt(bs, pr_len):
    path_500 ="prompt_500.json"
    path_1000 = "prompt_1000.json"
    path_2000 = "prompt_2000.json"
    if pr_len == 500:
        path = path_500
    elif pr_len == 1000:
        path = path_1000
    elif pr_len == 2000:
        path = path_2000
    prompts = [] 
    with open(path, "r") as f:
        prompts = json.load(f)
    return prompts[:bs]


def with_agent(engine,args, prompts):
    warm_up(engine)
    print(f"finish warm up")
    max_token = args.max_tokens
    agent_tokens = args.agent_tokens
    bs  = args.batch_size
    sp1 = """
    you are very powerful AI, Please help answer the below question and generate at least {max_token} tokens.
    """
    sp2 ="""
    Meta creates you and you are very good at answering the questinons, help me answer the below question and generate at least {max_token} tokens.
    """
    info = dict() #str -> ReactReq
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token) #this can be changed
    discard_sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1) #used for agent paralllelism 
    req_acts = dict()
    num_act = args.num_act
    now = time.time()
    reqs = []
    for i in range(args.batch_size):
        reqs.append([str(i), prompts[i]])
    finished = [] 
    print(f"1 with agent len(reqs):{len(reqs)}")
    while True:
        now = time.time()
        while reqs:
            now  = time.time()
            rid, prompt = reqs.pop()
            if 'i' not in rid:
                rid1 = rid+ "i0"
                rid2 = rid + "i1"
            else:
                rid1 = rid[:-2] + "i0"
                rid2 = rid[:-2] + "i1"
            if rid1 not in req_acts:
                req_acts[rid1] = 1
            if rid2 not in req_acts:
                req_acts[rid2] = 0
            req_num_act = req_acts[rid1] + req_acts[rid2]
            print(f"0 rid:{rid} and rid1:{rid1} and rid2:{rid2}")
            print(f"0.2 req_num_act:{req_num_act} and num_act:{num_act}")
            if rid1 not in info:
                req = ReactReq( arr1=now,arr2=now, rid1=rid1, rid2=rid2)
                req.r1_user_prompt = prompt
                info[rid1] = req
                info[rid2] = req
            else:
                #update the arr1 or arr2
                if req_num_act % 2 == 1:
                    info[rid1].arr1 = now
                else:
                    info[rid2].arr2 = now
            print(f"1 with_agent req_num_act:{req_num_act} and num_act:{num_act}")
            if req_num_act % 2 == 1:
                engine.add_request(request_id = rid1, inputs = sp1+ prompt, params = sampling_params, arrival_time = now)
                #do prefill for req2 
                engine.add_request(request_id = rid2, inputs = sp2+ prompt, params = discard_sampling_params, arrival_time = now)
            else:
                engine.add_request(request_id = rid2, inputs = sp2+ prompt, params = sampling_params, arrival_time = now)
                #do prefill for req1
                engine.add_request(request_id = rid1, inputs = sp1+ prompt, params = discard_sampling_params, arrival_time = now)
            try:
                request_outputs = engine.step()
            except Exception as e:
                print(f"error: {e}")
            for request_output in request_outputs:
                req_id = request_output.request_id
                rid1 = info[req_id].rid1
                rid2 = info[req_id].rid2
                print(f"1.2 with_agent, req_id:{req_id} and rid1:{rid1} and rid2:{rid2}")
                req_num_act = req_acts[rid1] + req_acts[rid2]
                output_text_len = len(request_output.outputs[0].token_ids)
                output_text = request_output.outputs[0].text
                print(f"1.3 req_num_act:{req_num_act} and num_act:{num_act}")
                if not request_output.finished and req_num_act != num_act:
                    #use agent parallelism 
                    if req_num_act % 2 == 0:
                        #req1 agent prefill, req2 prefill+ decode
                        if output_text_len % agent_tokens == 0:
                            #add rid1 into the engine
                            reqs.append([rid1, info[rid1].r2_user_prompt + output_text])
                    else:
                        #req2 agent prefill, req1 prefill+decode
                        if output_text_len % agent_tokens == 0:
                            #add rid2 into the engine
                            reqs.append([rid2, info[rid2].r1_user_prompt + output_text])
                elif request_output.finished and req_num_act == num_act:
                    #also use agent parallelism 
                    if req_num_act % 2 == 0:#r2 finished 
                        #r2 decode+prefill, r1 only prefill
                        reqs.append([rid1, info[rid1].r2_user_prompt + output_text])
                        info[rid1].total_duration += now - info[req_id].arr2
                        info[rid1].total_token += len(request_output.outputs[0].token_ids)
                        info[rid1].r1_user_prompt = info[rid1].r2_user_prompt + output_text
                        req_acts[rid1] += 1
                    else: #r1 finished
                        #r1 decode+prefill, r2 only prefill
                        reqs.append([rid2, info[rid2].r1_user_prompt + output_text])
                        info[rid1].total_duration += now - info[req_id].arr1
                        info[rid1].total_token += len(request_output.outputs[0].token_ids)
                        info[rid1].r2_user_prompt = info[rid1].r1_user_prompt + output_text
                        req_acts[rid2] += 1
                elif request_output.finished and req_num_act == num_act: #finish the react application
                    if req_num_act % 2 == 0:
                        info[rid1].total_duration += now - info[rid1].arr2
                        info[rid1].total_token += len(request_output.outputs[0].token_ids)
                    else:
                        info[rid1].total_duration += now - info[rid1].arr1
                        info[rid1].total_token += len(request_output.outputs[0].token_ids)
                    finished.append({
                        "request_id": rid,
                        "total_duration": info[rid1].total_duration,
                        "total_token": info[rid1].total_token
                    })
                    print(f"finished[-1]:{finished[-1]} and rid:{rid}")
        if not (reqs or engine.has_unfinished_requests()):
            break
        # print(f"2 engine has unfinished requests:{engine.has_unfinished_requests()}")    
    latencies = 0
    total_tokens = 0
    for d in finished:
        latencies += d["total_duration"]
        total_tokens += d["total_token"]
    normalized_latency = latencies/total_tokens
    avg_latency = latencies/len(finished)
    print(f"batch:{bs} and normalized_latency:{normalized_latency}")
    print(f"batch:{bs} and avg e2e latency:{avg_latency}")



def without_agent(engine, args,prompts):
    warm_up(engine)

    max_token = args.max_tokens
    bs  = args.batch_size
    sp1 = """
    you are very powerful AI, Please help answer the below question and generate at least {max_token} tokens.
    """
    sp2 ="""
    Meta creates you and you are very good at answering the questinons, help me answer the below question and generate at least {max_token} tokens.
    """
    info = dict() #str -> ReactReq
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token) #this can be changed
    #discard_sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1) #this can be changed
    req_acts = dict()
    num_act = args.num_act
    now = time.time()
    reqs = []
    for i in range(args.batch_size):
        reqs.append([str(i), prompts[i]])
    finished = [] 
    print(f"1 without agent len(reqs):{len(reqs)}")
    while True:
        now = time.time()
        while reqs:
            now  = time.time()
            rid, prompt = reqs.pop()
            if 'i' not in rid:
                rid1 = rid+ "i0"
                rid2 = rid + "i1"
            else:
                rid1 = rid[:-2] + "i0"
                rid2 = rid[:-2] + "i1"
            print(f"0 rid:{rid} and rid1:{rid1} and rid2:{rid2}")
            if rid1 not in req_acts:
                req_acts[rid1] = 1
            if rid2 not in req_acts:
                req_acts[rid2] = 0
            req_num_act = req_acts[rid1] + req_acts[rid2]
            if rid1 not in info:
                req = ReactReq( arr1=now, rid1=rid1, rid2=rid2)
                req.r1_user_prompt = prompt
                info[rid1] = req
                info[rid2] = req
                #update the arr1 or arr2
            if req_num_act % 2 == 1:
                info[rid1].arr1 = now
            else:
                info[rid2].arr2 = now
            add_rid = ""
            if req_num_act % 2 == 1:
                engine.add_request(request_id = rid1, inputs = sp1+ prompt, params = sampling_params, arrival_time = now)
                add_rid = rid1
            else:
                engine.add_request(request_id = rid2, inputs = sp2+ prompt, params = sampling_params, arrival_time = now)
                add_rid = rid2
            print(f"1.2 engine has add request and add_rid:{add_rid}")
        try:
            request_outputs = engine.step()
        #don't use agent parallelism 
        except Exception as e:
            print(f"error: {e}")
        #for request_output in request_outputs:
        for request_output in request_outputs:
            #this req has finished
            if request_output.finished: 
                print(f"1.5 request_output.finished")
                now = time.time()
                rid = request_output.request_id 
                rid1 = info[rid].rid1
                rid2 = info[rid].rid2
                init_rid = rid1[:-2]
                print(f"1.6 rid:{rid} and rid1:{rid1} and rid2:{rid2} and init_rid:{init_rid}")
                output_text = request_output.outputs[0].text
                rid2_num_act = req_acts[rid2]
                rid1_num_act = req_acts[rid1]
                rid_num_act = rid1_num_act + rid2_num_act
                output_len = len(request_output.outputs[0].token_ids)
                print(f"1.7 rid_num_act:{rid_num_act} and num_act:{num_act} and output_len:{output_len}")
                if rid_num_act != num_act:
                    if rid_num_act % 2 == 1:
                        info[rid].total_duration += now - info[rid].arr1
                        info[rid].total_token += output_len
                        #add rid2 into the engine
                        info[rid].r2_user_prompt = info[rid].r1_user_prompt + output_text
                        reqs.append([rid2, info[rid].r2_user_prompt ])
                        req_acts[rid2] = rid2_num_act + 1
                    else:
                        info[rid].total_duration += now - info[rid].arr2
                        info[rid].total_token += output_len
                        info[rid].r1_user_prompt = info[rid].r2_user_prompt + output_text 
                        reqs.append([rid1,info[rid].r1_user_prompt ])
                        req_acts[rid1] = rid1_num_act + 1
                else:
                    if rid_num_act % 2 == 1:
                        info[rid].total_duration += now - info[rid].arr1
                        info[rid].total_token += output_len
                    else:
                        info[rid].total_duration += now - info[rid].arr2
                        info[rid].total_token += output_len
                    finished.append({
                        "request_id": init_rid,
                        "total_duration": info[rid].total_duration,
                        "total_token": info[rid].total_token
                    })
                    print(f"finished[-1]:{finished[-1]} and rid:{rid}")
        if not (reqs or engine.has_unfinished_requests()):
            break
    print(f"len(finished):{len(finished)}")
    latencies = 0
    total_tokens = 0
    for d in finished:
        latencies += d["total_duration"]
        total_tokens += d["total_token"]
    if total_tokens !=0:
        normalized_latency = latencies/total_tokens
        avg_latency = latencies/len(finished)
        print(f"batch:{bs} and normalized_latency:{normalized_latency}")
        print(f"batch:{bs} and avg e2e latency:{avg_latency}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark.')

    parser.add_argument('--model', type=str, default='/data/Meta-Llama-3.1-8B-Instruct', help='model name')
    # parser.add_argument('--model-name', type=str, default='llama3_8B', help='engine name')
    # parser.add_argument('--request-rate', type=float, help='reqs/sec', default=5)
    # parser.add_argument('--tensor-parallel-size', type=int, help='tp size', default=1)
    parser.add_argument('--num-act', type=int, help='number of react steps', default=3)
    parser.add_argument('--batch-size', type=int, help='batch size', default=1)
    parser.add_argument('--agent-parallelism', action='store_true', help='whether use agent parallelism or not')
    parser.add_argument("--agent-tokens", type=int, help="number of tokens for agent parallelism", default=32)
    parser.add_argument("--max-tokens", type=int, help="max tokens for one llm call", default=128)
    # parser.add_argument('--preempt', type=bool, help='preempt', default=False)
    # parser.add_argument('--duration', type=int, help='duration in seconds', default=5)
    args = parser.parse_args() 
    print(f"args.agent_parallelism:{args.agent_parallelism}")       
    model = args.model 
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args)    
    prompts = get_prompt(args.batch_size, 500)
    #main(engine,args)
    if args.agent_parallelism:
        with_agent(engine, args, prompts)
    else:
        without_agent(engine, args, prompts)