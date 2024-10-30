import time 
import pickle
import argparse
#from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import json 
import time 

class ReactReq:
    def __init__(self, arr1=None, arr2=None, prompt_len=0, output_len=0,rid1= None, rid2=None, user_prompt = None, num_react=0):
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
        self.need_r1_react_num = num_react // 2 + 1
        self.need_r2_react_num = num_react // 2
        # print(f"need_r1_react_num:{self.need_r1_react_num} and need_r2_react_num:{self.need_r2_react_num}")
        self.r1_react_num = 0
        self.r2_react_num = 0
        self.react_num = num_react 
    def terminate_application(self):
        return self.r1_react_num + self.r2_react_num > self.react_num

    def final_terminate(self):
        return self.r1_react_num + self.r2_react_num == self.react_num
    
    def get_react_num(self):
        return self.r1_react_num + self.r2_react_num


class FinishReq:
    def __init__(self, rid1,rid2, output_text:str, output_len:int, now = None):
        self.rid1 = rid1
        self.rid2 = rid2
        self.output_text = output_text  
        self.output_len = output_len
        self.r0_finished = False
        self.r1_finished = False
        self.now = now 


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
    
def get_prompt(bs, pr_len):
    path_300 = "prompt_300.json"
    path_400 = "prompt_400.json"
    path_500 ="prompt_500.json"
    path_1000 = "prompt_1000.json"
    path_2000 = "prompt_2000.json"
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
    prompts = [] 
    with open(path, "r") as f:
        prompts = json.load(f)
    return prompts[1:bs+1]

def with_agent_optimized(args, prompts):
    print(f"start with agent")
    model = args.model 
    print(f"model:{model}")
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    warm_up(engine)   
    max_token = args.max_tokens
    agent_tokens = args.agent_tokens
    bs = args.batch_size
    print(f"max_token:{max_token} and agent_tokens:{agent_tokens}")
    sp1 = """
    you are very powerful AI, Please help answer the below question.
    """
    sp2 ="""
    Meta creates you and you are very good at answering the questinons, help me answer the below question.
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
    time.sleep(5)
    marked = dict()
    finished_reqs = dict() 
    maked_id = dict()
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
            if rid1 not in info:#first time react
                req_num_act = 1
            else:
                req_num_act = info[rid1].get_react_num()
            #print(f"0 rid:{rid} and rid1:{rid1} and rid2:{rid2} and req_num_act:{req_num_act}")
     
            if rid1 not in info:
                #print(f"0.5 rid1:{rid1} and rid2 not in info and req_num_act:{req_num_act}")
                req = ReactReq( arr1=now,arr2=None, rid1=rid1, rid2=rid2,num_react = num_act)
                req.r1_user_prompt = prompt
                req.r1_react_num =1#first time react 
                info[rid1] = req
                info[rid2] = req
            else:
                #update the arr1 or arr2
                #print(f"0.7 rid1:{rid1} and rid2 in info and req_num_act:{req_num_act}")
                if req_num_act % 2 == 1:
                    info[rid2].arr1 = now
                    info[rid1].arr1 = now
                elif req_num_act % 2 == 0:
                    info[rid1].arr2 = now
                    info[rid2].arr2 = now


            is_terminate = info[rid1].terminate_application()
            final_terminate = info[rid1].final_terminate()
            #print(f"1.5 with_agent rid:{rid} and req_num_act:{req_num_act} rid1={rid1} and rid2={rid2} ")
            #print(f"2 with_agent req_num_act:{req_num_act} and num_act:{num_act} and rid1={rid1} and rid2={rid2} and is_terminate:{is_terminate} and final_terminate:{final_terminate}")
            if req_num_act % 2 == 1 and not is_terminate:
                #do prefill+docode for req1
                engine.add_request(request_id = rid1, inputs = sp1+ prompt, params = sampling_params, arrival_time = now)
                # do prefill for req2 
                if not final_terminate:
                    engine.add_request(request_id = rid2, inputs = sp2+ prompt, params = discard_sampling_params, arrival_time = now)
                    #print(f"3 rid1:{rid1} is agent prefill and rid2:{rid2} is prefill+decode")
                    marked[rid2] = False 

            elif req_num_act % 2 == 0  and not is_terminate:
                #do prefill+decode for req2
                engine.add_request(request_id = rid2, inputs = sp2+ prompt, params = sampling_params, arrival_time = now)
                #do prefill for req1
                #info[rid1].r2_react_num += 1
                if not final_terminate:
                    engine.add_request(request_id = rid1, inputs = sp1+ prompt, params = discard_sampling_params, arrival_time = now)
                    #print(f"4 rid1:{rid1} is agent parallelism and rid2:{rid2} is prefill+decode")
                    marked[rid1] = False
       
        try:
            start = time.time()
            request_outputs = engine.step()
            end = time.time()
            print(f"4.2 one step time:{end-start}")
        except Exception as e:
            print(f"error: {e}")

        #print(f"4.5 len(request_outputs):{len(request_outputs)}")
        now = time.time()
        for request_output in request_outputs:
            req_id = request_output.request_id
            rx0 = info[req_id].rid1
            rx1 = info[req_id].rid2
            rid = rx0[:-2]
            init_id = req_id[:-2]
            print(f"0 with_agent, req_id:{req_id} and rx0:{rx0} and rx1:{rx1} and rid:{rid} and init_id:{init_id}")
            if num_act % 2 == 0:
                terminate_rid = rx1
            else:
                terminate_rid = rx0
            req_num_act = info[rx0].get_react_num()
                        
            output_text_len = len(request_output.outputs[0].token_ids)
            is_terminate = info[rx0].terminate_application()
            final_terminate = info[rx1].final_terminate()
            output_text = request_output.outputs[0].text
            #print(f"5 with_agent, req_id:{req_id} and rx0:{rx0} and rx1:{rx1} and req_num_act:{req_num_act} and req.finished:{request_output.finished} and o_len:{output_text_len} and is_terminate:{is_terminate}")
            #if req_num_act % 2 == 1, rx0 prefill+decode, rx1 only prefill,
            #if req_num_act % 2 == 0, rx1 prefill+decode, rx0 only prefill
            if not request_output.finished and not is_terminate:
                #analyze
                if req_num_act % 2 == 1: #rx0 prefill + decode, rx1 prefill
                    #print(f"6.8 req_id:{req_id} and rx0:{rx0} prefill+decode and rx1:{rx1} prefill and marked[x1]:{marked[rx1]}") 
                    #rx0 prefill+decode
                    #1)rx0 prefill+decode 完成一个iteration, 同时rx1完成了prefill,然后rx0生成的token数直接满足条件，那就继续让rx1 prefill，marked[rx1] = False
                    #2)rx0 prefill+decode 完成一个iteration, 同时rx1完成了prefill,然后rx0生成的token数不满足条件，那就继续让rx0 prefill+decode
                    #3)rx0 prefill+decode 完成一个iteration, 但是rx1没有完成prefill, 存下rx0的output_text, 
                    if req_id == rx0:#rx0 prefill + decode run some iteration, 
                        continue 
                        # if marked[rx1]:#rx1 done prefill
                        #     if rx0 in finished_reqs:
                        #         fin_req = finished_reqs[rx0]
                        #         fin_req.output_text = output_text
                        #         fin_req.output_len = output_text_len
                        #         fin_req.now = now
                        #     else:
                        #         fin_req = FinishReq(rx0, rx1, output_text, output_text_len,now)
                        #         finished_reqs[rx1] = fin_req
                        #         finished_reqs[rx0] = fin_req
                        #         print(f"6.9 req_id:{req_id} rx0:{rx0} prefill+decode and rx1:{rx1} prefill generate fin_req")

                    elif req_id == rx1:#r1 prefill not done, but rx0 may finish prefill+decode 
                            continue 
                            
                elif req_num_act % 2 == 0: #rx0 prefill, rx1 prefill + decode 
                    #print(f"8.3 req_id:{req_id} rx0:{rx0} prefill and rx1:{rx1} prefill + decode")
                    #rx1 prefill+decode run some iteration
                    if req_id == rx1:
                        continue 
                        # if marked[rx0]:#rx0 done prefill
                        #     if rx1 in finished_reqs:
                        #         fin_req = finished_reqs[rx1]
                        #         fin_req.output_text = output_text
                        #         fin_req.output_len = output_text_len
                        #         fin_req.now = now
                        #     else:
                        #         fin_req = FinishReq(rx0, rx1, output_text, output_text_len,now)
                        #         finished_reqs[rx1] = fin_req
                        #         finished_reqs[rx0] = fin_req
                        #         print(f"8.6 req_id:{req_id} rx0:{rx0} prefill and rx1:{rx1} prefill + decode generate fin_req")

                    elif req_id == rx0:#r0 not prefill done, but rx1 may finish prefill+decode r0->r1
                        continue


            elif request_output.finished and not is_terminate:
                #print(f"9 with_agent, req_id:{req_id} and rx0:{rx0} and rx1:{rx1} and req_num_act:{req_num_act} and request_output.finished and not is_terminate")
                if req_num_act % 2 == 1:#rx0 prefill + decode, r1 prefill
                    #print(f"9.4 req_id:{req_id} and rx0:{rx0} prefill + decode and rx1:{rx1} prefill and marked[rx1]:{marked[rx1]}")
                #1 rx0 prefill + decode, r1 prefill, it can decided by the req_num_act
                    #case 1:rx0 finished, rx1 not finished
                        #store/or update the fin_req, and then continue the rx1 prefill
                    #case 2:rx0 finished, rx1 finished
                        #rx0->prefill, rx0 + rx1 - > prefill + decode 
                    if req_id == rx0:#rx0 prefill+decode finished
                        #print(f"10 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1} prefill and marked[rx1]:{marked[rx1]} and req_num_act:{req_num_act} and num_act:{num_act}")
                        if marked[rx1] == True and req_num_act != num_act:#rx0 finished prefill+done, rx1 finished  prefill
                            #print(f"10.02 req_id:{req_id} and rx0:{rx0} ******* prefill+decode done and rx1:{rx1} agent prefill *****done")
                            if req_num_act != num_act:
                                reqs.append([rx1, info[rx0].r1_user_prompt + output_text])
                                # prompt = info[rx0].r1_user_prompt + output_text
                                #print(f"10.03 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1} ")
                                #print(f"10.1 req_id:{req_id} and rx0:{rx0} agent parallelism and rx1:{rx1} prefill+decode ")
                            info[rx0].r2_user_prompt = info[rx0].r1_user_prompt + output_text
                            info[rx0].total_duration += now - info[req_id].arr1 #Note: finish the r1 prefill+decode, statistics the duration
                            info[rx0].total_token += len(request_output.outputs[0].token_ids)#Note: finish the r1 prefill+decode, statistics the token
                            print(f"9 ****application id:{init_id} finish the r0:{rx0} for prefill+decode and r1:{rx1} for agent prefill  and the duration:{ now - info[req_id].arr1} and the token:{len(request_output.outputs[0].token_ids)} and total_duration:{info[rx1].total_duration} and info[rx1].total_token:{info[rx1].total_token} " )
                            #print(f"10.2 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1} prefill done and info[rx1].total_duration:{info[rx1].total_duration} and info[rx1].total_token:{info[rx1].total_token}")
                            info[rx1].r2_react_num += 1 #Note: finish the r1 prefill +decode and r2 agent prefill, now start to do r1 agent prefill and r2 prefill+decode
                            #print(f"10.3 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1} prefill done and after_req_num_act:{after_req_num_act} and num_act:{num_act}")
                        elif req_num_act != num_act:#rx1 not finished,store the rx0 until the rx1 finished,also the application is not a final terminate
                            if rx1 not in finished_reqs:
                                fin_req = FinishReq(rx0, rx1, output_text, output_text_len,now)
                                #print(f"10.25 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1} prefill generate fin_req")
                                finished_reqs[rx1] = fin_req
                                finished_reqs[rx0] = fin_req 
                                finished_reqs[rx0].r0_finished = True
                                finished_reqs[rx1].r0_finished = True
                            else:
                                #update the rx0 output_text and output_len and now 
                                finished_reqs[rx1].output_text = output_text
                                finished_reqs[rx1].output_len = output_text_len
                                finished_reqs[rx0].output_text = output_text
                                finished_reqs[rx0].output_len = output_text_len
                                finished_reqs[rx0].r0_finished = True
                                finished_reqs[rx1].r0_finished = True
                                finished_reqs[rx0].now = now
                                finished_reqs[rx1].now = now
                                #print(f"10.4 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1} prefill update fin_req")
                        elif req_num_act == num_act:#finish the application
                            #print(f"10.5 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1} prefill and req_num_act == num_act")
                            fin_req = FinishReq(rx0, rx1, output_text, output_text_len,now)
                            finished_reqs[rx1] = fin_req
                            finished_reqs[rx0] = fin_req 
                            info[rx0].r1_react_num += 1 #TODO(xiao):this may have bug, bc it can only work when the num_act is even

                    elif req_id == rx1:#rx1 agent prefill finished
                        if marked[rx1] == False:
                            #print(f"10.5 req_id:{req_id} and rx0:{rx0} prefill + decode  rx1:{rx1} agent agent prefill***** done")
                            marked[rx1] = True
                        #rx0  finished
                        #print(f"11 req_id:{req_id} and rx0:{rx0} prefill + decode and rx1:{rx1} prefill done and rx0_stored:{rx0_stored}")
                        if rx0 in finished_reqs:
                            fin_req = finished_reqs[rx0]
                            if fin_req.r0_finished:
                                #print(f"11.6 req_id:{req_id} rx0:{rx0} prefill + decode and rx1:{rx1} prefill done and rx0_stored:{rx0_stored}")
                                if req_num_act != num_act:
                                    reqs.append([rx1, info[rx0].r1_user_prompt + output_text])
                                    # prompt = info[rx1].r1_user_prompt + output_text
                                    #print(f"11.7 req_id:{req_id} and rx0:{rx0} prefill + decode done and rx1:{rx1}")
                                #print(f"11.7 req_id:{req_id} rx0:{rx0} convert to agent prefill  and rx1:{rx1} convert prefill+decode")
                                #print(f"11.7 req_id:{req_id} rx0:{rx0} prefill + decode ******done and rx1:{rx1} still agent prefill done")
                                info[rx0].total_duration += now - info[rx0].arr1
                                info[rx0].total_token += fin_req.output_len
                                #print(f"11.8 req_id:{req_id} rx0:{rx0} prefill + decode done and rx1:{rx1} prefill done and info[rx0].total_duration:{info[rx0].total_duration} and info[rx0].total_token:{info[rx0].total_token}")
                                info[rx0].r2_user_prompt = info[rx0].r1_user_prompt + output_text
                                info[rx0].r2_react_num += 1
                                marked[rx0] = False
                                print(f"10 ****application id:{init_id} finish the r0:{rx0} for prefill+decode and r1:{rx1} for agent prefill  and the duration:{ now - info[rx0].arr1} and the token:{fin_req.output_len} and total_duration:{info[rx1].total_duration} and info[rx1].total_token:{info[rx1].total_token} " )
                                #print(f"11.9 req_id:{req_id} rx0:{rx0} prefill done and rx1:{rx1} prefill + decode done and after_rect_num:{after_rect_num} and num_act:{num_act}")
                                # finished_reqs.remove(rx0)
                                # finished_reqs.remove(rx1)
                                if rx0 in finished_reqs:
                                    finished_reqs.pop(rx0)
                                if rx1 in finished_reqs:
                                    finished_reqs.pop(rx1)
                            #reqs.append([rx0, fin_req.prompt + output_text,])
                elif req_num_act %2 == 0:#r0 prefill, rx1 prefill + decode
                    #print(f"11.5 req_id:{req_id} and rx0:{rx0} prefill and rx1:{rx1} prefill + decode req_num_act %2 == 0")
                    if req_id == rx1:#r1 prefill+decode finished
                        if marked[rx0] == True:#r0 prefill finished, r1 prefill+decode finished
                            if req_num_act != num_act:
                                reqs.append([rx1, info[rx0].r2_user_prompt + output_text])
                            # marked[rx0] = False
                            info[rx0].r1_user_prompt = info[rx0].r2_user_prompt + output_text
                            info[rx0].total_duration += now - info[req_id].arr2
                            info[rx0].total_token += len(request_output.outputs[0].token_ids)
                            info[rx0].r1_react_num += 1
                            print(f"11 ****application id:{init_id} finish the r0:{rx0} for agent prefill and r1:{rx1} for prefill+ decode and the duration:{ now - info[req_id].arr2} and the token:{len(request_output.outputs[0].token_ids)} and total_duration:{info[rx1].total_duration} and info[rx1].total_token:{info[rx1].total_token} " )

                            #print(f"11.6 req_id:{req_id} and rx0:{rx0} prefill done and rx1:{rx1} prefill + decode done and info[rx1].total_duration:{info[rx1].total_duration} and info[rx1].total_token:{info[rx1].total_token}")
                        else:#r0 not finished
                            if rx0 not in finished_reqs:
                                fin_req = FinishReq(rx0, rx1, output_text, output_text_len,now)
                                #print(f"12 req_id:{req_id} and rx0:{rx0} prefill and rx1:{rx1} prefill + decode and rx0 not in finished_reqs")
                                finished_reqs[rx0] = fin_req
                                finished_reqs[rx1] = fin_req
                                finished_reqs[rx0].r1_finished = True
                                finished_reqs[rx1].r1_finished = True
                            else:
                                finished_reqs[rx0].output_text = output_text
                                finished_reqs[rx0].output_len = output_text_len
                                finished_reqs[rx1].output_text = output_text
                                finished_reqs[rx1].output_len = output_text_len
                                finished_reqs[rx0].now = now
                                finished_reqs[rx1].now = now
                                finished_reqs[rx0].r1_finished = True
                                finished_reqs[rx1].r1_finished = True
                                #print(f"12.1 req_id:{req_id} and rx0:{rx0} prefill and rx1:{rx1} prefill + decode and rx0 in finished_reqs")
                    elif req_id == rx0:#r0 finished
                        #print(f"12.5 req_id:{req_id} and rx0:{rx0} agent prefill done and rx1:{rx1} prefill + decode and marked[rx0]:{marked[rx0]}")
                        if marked[rx0] == False:
                            marked[rx0] = True
                        #rx1  finished
                        if rx1 in finished_reqs:
                            fin_req = finished_reqs[rx1]
                            if fin_req.r1_finished:
                                if req_num_act != num_act:
                                    reqs.append([rx0, info[rx0].r2_user_prompt + output_text])
                                    #print(f"12.7 req_id:{req_id} rx0:{rx0} convert prefill+decode and rx1:{rx1} may convert agent prefill")
                                marked[rx1] = False
                                #print(f"12.75 req_id:{req_id} rx0:{rx0} prefill done and rx1:{rx1} prefill + decode done and and info[rx0].total_duration:{info[rx0].total_duration} and info[rx0].total_token:{info[rx0].total_token} ")
                                info[rx0].total_duration += now - info[rx0].arr2
                                info[rx0].total_token += fin_req.output_len
                                info[rx0].r1_user_prompt = info[rx0].r2_user_prompt + output_text
                                info[rx0].r1_react_num += 1
                                print(f"12 ****application id:{init_id} finish the r0:{rx0} for agent prefill and r1:{rx1} for prefill+ decode and the duration:{ now - info[req_id].arr2} and the token:{fin_req.output_len} and total_duration:{info[rx1].total_duration} and info[rx1].total_token:{info[rx1].total_token} " )
                                #print(f"12.8 req_id:{req_id} rx0:{rx0} prefill done and rx1:{rx1} prefill + decode done and info[rx0].total_duration:{info[rx0].total_duration} and info[rx0].total_token:{info[rx0].total_token}")
                                if rx0 in finished_reqs:
                                    finished_reqs.pop(rx0)
                                if rx1 in finished_reqs:
                                    finished_reqs.pop(rx1)
                elif request_output.finished and is_terminate:
                    #print(f"13 with_agent, req_id:{req_id} and rx0:{rx0} and rx1:{rx1} and req_num_act:{req_num_act} and req.finished:{request_output.finished} and o_len:{output_text_len} and terminate_rid:{terminate_rid}")
                    if req_num_act % 2 == 1:
                        if req_id == rx0:
                            info[rx1].r1_react_num += 1
                        
                    elif req_num_act % 2 ==0:
                        if req_id == rx1:
                            info[rx0].r2_react_num += 1

        for request_output in request_outputs:
            req_id = request_output.request_id
            rx0 = info[req_id].rid1
            rx1 = info[req_id].rid2
            rid = rx0[:-2]
            if num_act % 2 == 0:
                terminate_rid = rx1
            else:
                terminate_rid = rx0
            req_num_act = info[rx0].get_react_num()
            #print(f"17 with_agent, req_id:{req_id} and rx0:{rx0} and rx1:{rx1} and req_num_act:{req_num_act} and req.finished:{request_output.finished} and o_len:{output_text_len} and terminate_rid:{terminate_rid}")
            if request_output.finished and req_num_act == num_act+1:
                #finish the application
                if req_num_act % 2 == 0 and req_id == terminate_rid and rid not in maked_id:
                    #print(f"13 rid2 finish the application req_num_act:{req_num_act} and num_act:{num_act} and terminate_rid:{terminate_rid}")
                    fin_req = finished_reqs[rx0]
                    info[rx0].total_duration += fin_req.now - info[rx0].arr1 #TODO(bug)
                    info[rx0].total_token += len(request_output.outputs[0].token_ids)
                    finished.append({
                        "request_id": rid,
                        "total_duration": info[rx0].total_duration,
                        "total_token": info[rx0].total_token
                    })
                    maked_id[rid] = True
                    print(f"14 finished[-1]:{finished[-1]} and rid:{rid} and terminate_rid:{terminate_rid} ")
                elif req_num_act % 2 == 1 and req_id == terminate_rid and rid not in maked_id:
                    fin_req = finished_reqs[rx0]
                    info[rx0].total_duration += fin_req.now - info[rx0].arr2 #TODO(bug)
                    info[rx0].total_token += len(request_output.outputs[0].token_ids)
                    #print(f"15 rid1 finish the application req_num_act:{req_num_act} and num_act:{num_act} and terminate_rid:{terminate_rid} ")
                    finished.append({
                        "request_id": rid,
                        "total_duration": info[rx0].total_duration,
                        "total_token": info[rx0].total_token
                    })
                    maked_id[rid] = True
                    print(f"16 finished[-1]:{finished[-1]} and rid:{rid}")

        if not (reqs or engine.has_unfinished_requests()):
            break
        # print(f"2 engine has unfinished requests:{engine.has_unfinished_requests()}")    
    latencies = 0
    total_tokens = 0
    print(f"len(finished):{len(finished)} and bs:{bs}")
    normizliaed = 0 
    i = 0 
    for d in finished:
        if d["total_token"] != 0:
            latencies += d["total_duration"]
            total_tokens += d["total_token"]
            normizliaed += d["total_duration"]/d["total_token"]
            i += 1
    normalized_latency = normizliaed/i
    avg_latency = latencies/i
    print(f"batch:{bs} and normalized_latency:{normalized_latency}")
    print(f"batch:{bs} and avg e2e latency:{avg_latency}")


def without_agent(args,prompts):
    print(f"without_agent start with agent")
    model = args.model 
    print(f"model:{model}")
    engine_args = EngineArgs(model=model, enable_prefix_caching=True, enable_chunked_prefill=True)
    engine = LLMEngine.from_engine_args(engine_args) 
    warm_up(engine)   

    max_token = args.max_tokens
    bs  = args.batch_size
    print(f"without_agent max_token:{max_token} ")
    sp1 = """
    you are very powerful AI, Please help answer the below question.
    """
    sp2 ="""
    Meta creates you and you are very good at answering the questinons, help me answer the below question.
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
            if req_num_act % 2 == 1:
                engine.add_request(request_id = rid1, inputs = sp1+ prompt, params = sampling_params, arrival_time = now)
            else:
                engine.add_request(request_id = rid2, inputs = sp2+ prompt, params = sampling_params, arrival_time = now)
            #print(f"1.2 engine has add request and add_rid:{add_rid}")
        
        try:
            start = time.time()
            request_outputs = engine.step()
            end = time.time()
            print(f"one step time:{end-start}")
        #don't use agent parallelism 
        except Exception as e:
            print(f"error: {e}")
        #for request_output in request_outputs:
        for request_output in request_outputs:
            #this req has finished
            if request_output.finished: 
                #print(f"1.5 request_output.finished")
                now = time.time()
                rid = request_output.request_id 
                rid1 = info[rid].rid1
                rid2 = info[rid].rid2
                init_rid = rid1[:-2]
                #print(f"1.6 rid:{rid} and rid1:{rid1} and rid2:{rid2} and init_rid:{init_rid}")
                output_text = request_output.outputs[0].text
                rid2_num_act = req_acts[rid2]
                rid1_num_act = req_acts[rid1]
                rid_num_act = rid1_num_act + rid2_num_act
                output_len = len(request_output.outputs[0].token_ids)
                #print(f"1.7 rid_num_act:{rid_num_act} and num_act:{num_act} and output_len:{output_len}")
                if rid_num_act != num_act:
                    if rid_num_act % 2 == 1: #rid == rid1
                        info[rid].total_duration += now - info[rid].arr1
                        info[rid].total_token += output_len
                        #add rid2 into the engine
                        info[rid].r2_user_prompt = info[rid].r1_user_prompt + output_text
                        reqs.append([rid2, info[rid].r2_user_prompt ])
                        print(f"12 ****application id:{init_rid} finish the r0:{rid1} for prefill+decode and r1:{rid2} and the duration:{ now - info[rid].arr1} and the token:{output_len} and total_duration:{info[rid].total_duration} and info[rx1].total_token:{info[rid].total_token} " )

                        req_acts[rid2] = rid2_num_act + 1
                    else:
                        info[rid].total_duration += now - info[rid].arr2
                        info[rid].total_token += output_len
                        info[rid].r1_user_prompt = info[rid].r2_user_prompt + output_text 
                        reqs.append([rid1,info[rid].r1_user_prompt ])
                        print(f"13 ****application id:{init_rid} finish the r0:{rid1}  and r1:{rid2} prefil+decode and the duration:{ now - info[rid].arr2} and the token:{output_len} and total_duration:{info[rid].total_duration} and info[rx1].total_token:{info[rid].total_token} " )

                        req_acts[rid1] = rid1_num_act + 1
                else:
                    if rid_num_act % 2 == 1:
                        info[rid].total_duration += now - info[rid].arr1
                        info[rid].total_token += output_len
                        print(f"14 ****application id:{init_rid} finish the r0:{rid1} for prefill+decode and r1:{rid2} and the duration:{ now - info[rid].arr1} and the token:{output_len} and total_duration:{info[rid].total_duration} and info[rx1].total_token:{info[rid].total_token} " )

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
        #print(f"2 engine has unfinished requests:{engine.has_unfinished_requests()}")
    print(f"len(finished):{len(finished)}")
    latencies = 0
    total_tokens = 0
    normizliaed =0
    i = 0
    for d in finished:
        if d["total_token"] != 0:
            normizliaed += d["total_duration"]/d["total_token"]
            latencies += d["total_duration"]
            i += 1
    if i !=0:
        normalized_latency = normizliaed /i
        avg_latency = latencies/i
        print(f"batch:{bs} and normalized_latency:{normalized_latency}")
        print(f"batch:{bs} and avg e2e latency:{avg_latency}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark.')

    parser.add_argument('--model', type=str, default='/home/xiaoxiang/data/Llama-3.1-8B', help='model name')
    parser.add_argument('--num-act', type=int, help='number of react steps', default=3)
    parser.add_argument('--batch-size', type=int, help='batch size', default=1)
    parser.add_argument('--agent-parallelism', action='store_true', help='whether use agent parallelism or not')
    parser.add_argument("--agent-tokens", type=int, help="number of tokens for agent parallelism", default=64)
    parser.add_argument("--max-tokens", type=int, help="max tokens for one llm call", default=256)
    args = parser.parse_args() 
    print(f"args.agent_parallelism:{args.agent_parallelism}")      
    prompts = get_prompt(args.batch_size, 500)
    if args.agent_parallelism:
        print(f"start with agent")
        with_agent_optimized(args, prompts)
    else:
        without_agent( args, prompts)