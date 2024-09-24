from generate_trace import  generate_vllm_text_requests_fixed
from vllm import LLM, SamplingParams
import time 
import pickle
import argparse
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from transformers import AutoTokenizer
import csv, os 
def warmup(engine, sampling_params):

    prompt ="Metaphorical language has been summoned to describe the many enigmatic addressing modes of the instructions at hand. The speakers have resorted to grandiose expressions to convey their immense wonder and admiration for the sheer functionality of the aforementioned directives. Among the mystifying commands are the enigmatic JMP ABCD, the confounding MOV AX, [BX+SI], the inscrutable MOV AX, [100], the cryptic MOV AX, [BX], the perplexing MOV AX, [BX\*2+SI], the unfathomable MOV AX, BX, and finally, the enigmatic MOV AX, 7."

    prompts = []

    prompts.append(prompt)

    for i in range(3):
        engine.add_request(request_id = i, inputs = prompt, params = sampling_params, arrival_time = time.time())
    while True:
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output.outputs[0].text)
        if not engine.has_unfinished_requests(): 
            break

class React:
    def __init__(self, prompt_len:int, output_len:int, arrival_time, finished_time, schedule_time=None):
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.arrival_time = arrival_time  
        self.finished_time = finished_time 
        self.schedule_time = schedule_time

def main(engine,request_rate:int, requests:list,args):
    #warm up
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=512) #this can be changed
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    warmup(engine, sampling_params)
    print(f"************warmup finished************")
    i = 0
    finished = [] 
    num_act = int(args.num_act)
    total_latency = [] 
    metadata = dict()
    freq = dict() 
    finished_metadata = dict()
    start = time.time()
    for req in requests:
        req[1] = start + req[1]
    end = time.time()
    print(f"end - start:{end - start}")
    
    while True:
        now = time.time()
        while requests:
            now = time.time()
            if requests[0][1] <= now:
                rid, request_time, whole_prompt = requests.pop(0)
                if type(rid) == str:
                    arrival_time = request_time
                    metadata[rid] = [whole_prompt, arrival_time,request_time]
                else:
                    arrival_time =  request_time
                    metadata[rid] = [whole_prompt, arrival_time,request_time]
                print(f"engine add_request rid:{rid}  and request_time:{request_time} and now:{now} ")
                engine.add_request(request_id = rid, inputs = whole_prompt, params = sampling_params, arrival_time=request_time)
                if rid not in freq:
                    freq[rid] = 1
            else:
                break
        try:
            request_outputs: List[RequestOutput] = engine.step()
        except Exception as e:
            if len(finished) > 0:
                tmp_req = int(request_rate)
                out_path = f"./data/vllm_react_model_name_{args.model_name}_request_rate{tmp_req}_num_act{int(num_act)}.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump(finished, f)
                    print(f"benchmark finished, results saved to {out_path} exception")
                
        for request_output in request_outputs:
            if request_output.finished:
                finish_time  = time.time() 
                rid = str(request_output.request_id)

                
                if type(rid) == str:
                    info = metadata[rid]
                elif type(rid) == int:
                    info = metadata[str(rid)]
                arrival_time = info[1] # get the arrival time
                schedule_time = request_output.schedule_time
                print(f" request id:{rid} and schedule_time:{schedule_time} and arrival_time:{arrival_time}")
                old_num_act = freq[rid]
                print(f"request_output.request_id:{rid} and type(rid):{type(rid)} and duration:{finish_time - arrival_time} and act:{old_num_act}")
                if freq[rid] != num_act:
                    freq[rid] = freq[rid] + 1
                    whole_prompt = info[0] # get the whole prompt    
                    output = request_output.outputs[0].text            
                    new_prompt = whole_prompt + output  #update the whole prompt to simulate the react 
                    react = React(len(request_output.prompt_token_ids),  len(request_output.outputs[0].token_ids),arrival_time , finish_time, schedule_time)
                    if rid not in finished_metadata:
                        finished_metadata[rid] = [react]
                    else:
                        finished_metadata[rid].append(react)
                    new_arrival_time = time.time()
                    print(f"engine readd request rid:{rid} and new_arrival_time:{new_arrival_time} and duration:{finish_time - arrival_time} and vllm waiting time:{schedule_time- arrival_time}")
                    requests.append([rid, new_arrival_time, new_prompt])
                    #engine.add_request(request_id = rid, inputs = new_prompt, params = sampling_params, arrival_time=new_arrival_time)
                else:
                    #print(f"engine finish react rid:{rid} and arrival_time:{arrival_time}")
                    finished_time = finish_time
                
                    react = React(len(request_output.prompt_token_ids),  len(request_output.outputs[0].token_ids),arrival_time , finished_time, schedule_time)
                    print(f"engine finish react rid:{rid} and arrival_time:{arrival_time} and finished_time:{finished_time} and duration:{finished_time - arrival_time} and vllm waiting time:{schedule_time- arrival_time}")
                    reacts = finished_metadata[rid]
                    reacts.append(react)
                    assert len(reacts) == num_act
                    output_len = 0
                    duration = 0 
                    queue_duration = 0
                    for react in reacts:
                        output_len  = output_len +  react.output_len
                        arrival_time = react.arrival_time 
                        finished_time = react.finished_time 
                        diff = finished_time - arrival_time
                        duration = diff + duration
                        schedule_time = react.schedule_time
                        queue_duration += schedule_time - arrival_time
                        queue_time = schedule_time - arrival_time
                        print(f"*******rid:{rid} and output_len:{react.output_len} and diff:{diff} and output_len:{output_len} and duration:{duration}")
                        print(f"*******rid:{rid} and schedule_time:{schedule_time} and arrival_time:{arrival_time} and queue_time:{queue_time}")

                    if len(request_output.outputs) > 1:
                        for output in request_output.outputs:
                            print(f"output token ids:{len(output.token_ids)}")
                    if output_len== 0:
                        continue
                    per_token_latency = duration / output_len
                    finished.append({
                        "request_id": rid,
                        "duration":duration,
                        "generated_text_len": output_len,
                        "per_token_latency": per_token_latency,
                        "queue_duration":queue_duration
                    })
                    print(f"finished[-1]:{finished[-1]} and request:{rid} and duration:{duration} and per_token_latency:{per_token_latency} and queue_duration:{queue_duration}")
                    total_latency.append(per_token_latency)

            else:
                continue
        if len(requests) > 0:
            requests.sort(key=lambda x: x[1]) #TODO:may have problem 
        if not (requests or engine.has_unfinished_requests()):
            break
    normalized_latency = sum(total_latency) / len(total_latency)
    tmp_req = int(request_rate)
    print(f"request_rate:{tmp_req} and num_act:{int(num_act)} and normalized_latency:{normalized_latency} and len(total_latency):{len(total_latency)}")
    
    out_path = f"./data/vllm_react_model_name_{args.model_name}_request_rate{tmp_req}_num_act{int(num_act)}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(finished, f)
    print(f"benchmark finished, results saved to {out_path}")
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark.')

    parser.add_argument('--model', type=str, default='/data/Meta-Llama-3.1-8B-Instruct', help='model name')
    parser.add_argument('--model-name', type=str, default='llama3_8B', help='engine name')
    parser.add_argument('--request-rate', type=float, help='reqs/sec', default=5)
    parser.add_argument('--tensor-parallel-size', type=int, help='tp size', default=1)
    parser.add_argument('--num-act', type=int, help='number of active requests', default=3)
    # parser.add_argument('--preempt', type=bool, help='preempt', default=False)
    parser.add_argument('--duration', type=int, help='duration in seconds', default=5)

    args = parser.parse_args()

    request_rate = args.request_rate
    duration = args.duration
    model = args.model
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size > 1:
        engine_args = EngineArgs(model=model, enable_prefix_caching=True, tensor_parallel_size=tensor_parallel_size)
    else:
        engine_args = EngineArgs(model=model, enable_prefix_caching=True)
    engine = LLMEngine.from_engine_args(engine_args)
    requests = generate_vllm_text_requests_fixed(request_rate, duration)
    print(f"********requst_rate:{request_rate}, duration:{duration},  and model_name:{args.model_name}*******")
    main(engine, request_rate, requests, args)