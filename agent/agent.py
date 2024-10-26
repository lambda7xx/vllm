from vllm import LLM, SamplingParams
import time 
import pickle
import argparse
#from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import json 
import time 

class ReactReq:
    def __init__(self, arr1=None, arr2=None, prompt_len=0, output_len=0,rid1= None, rid2=None):
        self.arr1 = arr1
        self.arr2 = arr2
        self.prompt_len = prompt_len
        self.output_len = output_len 
        self.rid1 = rid1
        self.rid2 = rid2 #rid1 is the first request id, rid2 is the second request id
        self.total_duration = 0
        self.total_token = 0 

def warm_up(engine):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=25) #this can be changed

    
    prompt ="Metaphorical language has been summoned to describe the many enigmatic addressing modes of the instructions at hand. The speakers have resorted to grandiose expressions to convey their immense wonder and admiration for the sheer functionality of the aforementioned directives. Among the mystifying commands are the enigmatic JMP ABCD, the confounding MOV AX, [BX+SI], the inscrutable MOV AX, [100], the cryptic MOV AX, [BX], the perplexing MOV AX, [BX\*2+SI], the unfathomable MOV AX, BX, and finally, the enigmatic MOV AX, 7."
    for i in range(1):
        engine.add_request(request_id = i, inputs = prompt, params = sampling_params, arrival_time = time.time())
    while True:
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output is not None:
                #print(request_output.request_id, request_output.completion.text)
                print(request_output.outputs[0].text)
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

def main(engine, args):
    warm_up(engine)

    prompts = get_prompt(args.batch_size, 500)
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
    #discard_sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1) #this can be changed
    req_acts = dict()
    num_act = args.num_act
    now = time.time()
    reqs = []
    for i in range(args.batch_size):
        reqs.append([str(i)+"0", sp1+prompts[i]])
    finished = [] 
    while True:
        now = time.time()
        while reqs:
            now  = time.time()
            rid, prompt = reqs.pop()
            rid1 = rid[:-1] + "0"
            rid2 = rid[:-1] + "1"
            if rid1 not in info:
                req = ReactReq( arr1=now, rid1=rid1, rid2=rid2)
                info[rid1] = req
                info[rid2] = req
            else:
                #update the arr1 or arr2
                if rid == rid1:
                    info[rid1].arr1 = now
                else:
                    info[rid2].arr2 = now
            if rid1 not in req_acts:
                req_acts[rid1] = 1
            engine.add_request(request_id = rid, inputs = prompt, params = sampling_params, arrival_time = now)

        try:
            request_outputs = engine.step()
        #don't use agent parallelism 
        except Exception as e:
            print(f"error: {e}")

        #for request_output in request_outputs:
        if args.agent_parallelism:
            
        else:
            for request_output in request_outputs:
                #this req has finished
                if request_output.finished: 
                    now = time.time()
                    rid = request_output.request_id 
                    init_rid = rid[:-1] 
                    rid1 = info[rid].rid1
                    rid2 = info[rid].rid2
                    rid1_num_act = req_acts[rid1]
                    if rid2 not in req_acts:
                        rid2_num_act = 0
                    else :
                        rid2_num_act = req_acts[rid2]
                    rid_num_act = rid1_num_act + rid2_num_act
                    output = request_output.outputs[0].text
                    if rid_num_act != num_act:
                        if rid1 == rid:
                            info[rid].total_duration += now - info[rid].arr1
                            info[rid].total_token += len(output.token_ids)
                            #add rid2 into the engine
                            reqs.append([rid2, sp2+prompts[int(rid2[:-1])]])
                            req_acts[rid2] = rid2_num_act + 1
                        else:
                            info[rid].total_duration += now - info[rid].arr2
                            info[rid].total_token += len(output.token_ids)
                            reqs.append([rid1, sp1+prompts[int(rid1[:-1])]])
                            req_acts[rid1] = rid1_num_act + 1
                    else:
                        if rid1 == rid:
                            info[rid].total_duration += now - info[rid].arr1
                            info[rid].total_token += len(output.token_ids)
                        else:
                            info[rid].total_duration += now - info[rid].arr2
                            info[rid].total_token += len(output.token_ids)
                        finished.append({
                            "request_id": init_rid,
                            "total_duration": info[rid].total_duration,
                            "total_token": info[rid].total_token
                        })
                        print(f"finished[-1]:{finished[-1]} and rid:{rid}")
        if not engine.has_unfinished_requests():
            break
        latencies = 0
        total_tokens = 0
        for d in finished:
            latencies += d["total_duration"]
            total_tokens += d["total_token"]
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
    parser.add_argument('--agent-parallelism', type=bool, help='whether use agent parallelism or not', default=False)
    parser.add_argument("--agent-tokens", type=int, help="number of tokens for agent parallelism", default=16)
    parser.add_argument("--max-tokens", type=int, help="max tokens for one llm call", default=128)
    # parser.add_argument('--preempt', type=bool, help='preempt', default=False)
    # parser.add_argument('--duration', type=int, help='duration in seconds', default=5)
    args = parser.parse_args()        
    model = args.model 
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args)    
    main(engine,args)