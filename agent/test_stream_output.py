from vllm import LLM, SamplingParams
import time 
import pickle
import argparse
#from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
    # "AsyncLLMEngine",
    # "AsyncEngineArgs",


def main(engine):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=16) #this can be changed

    
    prompt ="Metaphorical language has been summoned to describe the many enigmatic addressing modes of the instructions at hand. The speakers have resorted to grandiose expressions to convey their immense wonder and admiration for the sheer functionality of the aforementioned directives. Among the mystifying commands are the enigmatic JMP ABCD, the confounding MOV AX, [BX+SI], the inscrutable MOV AX, [100], the cryptic MOV AX, [BX], the perplexing MOV AX, [BX\*2+SI], the unfathomable MOV AX, BX, and finally, the enigmatic MOV AX, 7."
    for i in range(1):
        engine.add_request(request_id = i, inputs = prompt, params = sampling_params, arrival_time = time.time())
    while True:
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output is not None:
                #print(request_output.request_id, request_output.completion.text)
                print(request_output.outputs[0].text, len(request_output.outputs[0].token_ids))
        if not engine.has_unfinished_requests():
            break   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark.')

    parser.add_argument('--model', type=str, default='/data/Meta-Llama-3.1-8B-Instruct', help='model name')
    # parser.add_argument('--model-name', type=str, default='llama3_8B', help='engine name')
    # parser.add_argument('--request-rate', type=float, help='reqs/sec', default=5)
    # parser.add_argument('--tensor-parallel-size', type=int, help='tp size', default=1)
    # parser.add_argument('--num-act', type=int, help='number of active requests', default=3)
    # # parser.add_argument('--preempt', type=bool, help='preempt', default=False)
    # parser.add_argument('--duration', type=int, help='duration in seconds', default=5)
    args = parser.parse_args()        
    model = args.model 
    engine_args = EngineArgs(model=model, enable_prefix_caching=True)
    engine = LLMEngine.from_engine_args(engine_args)    
    main(engine)