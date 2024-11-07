N1
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_token) #this can be changed
agent_prefill_sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1) #this can be changed
start = 0
agent_token = args.agent_token 
bs = args.bs
n1_need_act =  react_num // 2 + react_num % 2
finished_reqs = 0
counter =dict() 
while finished_reqs != bs
    if start == 0:
        start == 1
    else:
        data = recv_from_b()#recv data from B

        for d in data:
            rid = d.rid
            output_text = d.output_text
            output_len = d.output_len
            prompt = d.rid
            if d.finished:#it means the b is finished prefill+decode, so N1 do prefill + decode for this req
                reqs.append([rid, prompt+output_text, PREFILL+DECODE]
                #statistical the time
                info[rid].duration += now - info[rid].send_time
                info[rid].token += output_le
            else:
                if agent_prefill[rid] == True:#previous agent prefill for this rid is done
                    reqs.append([rid, prompt+output_text, AGENT_PREFILL])
                    agent_prefill[rid] == Flase 
                else:#previous agent prefill is not done
                    waiting_reqs.append([rid, prompt+output_text, AGENT_PREFILL])

    for req in reqs:
        req_type = req.req_type 
        if req_type == AGENT_PREFILL:
            use  agent_prefill_sampling_params for agent prefill
            
        else:
            use sampling_params   for prefill + decode 
            if counter[rid] < n1_need_act:
                counter[rid] +=1

    request_output = engine.step()
    send_data= dict()
    for req in request_output:
        rid = req.rid
        req_type  = req.req_type
        output_len = req.output_len
        if req.finished:
            if req_type = PREFILL+DECODE:
                #send the output text to N2
                if counter[rid] < n1_need_act:
                    send_data[rid] = output_text....
                    send_data[rid].finished= True
                    finished_seq += 1 
                #if counter[rid] == n1_need_act, this application is done 

                #statistical the time
                info[rid].duration += now - info[rid].arr
                info[rid].token += output_len
                info[rid].send_time = now #record the send time,because the N2 is for the next llm call

            else:
                agent_prefill[rid] = True
                if rid in waiting_req:
                    reqs.append(waiting_req[rid])
                    waiting_reqs.pop(rid)
        else:
            if req_type = PREFILL+DECODE:
                if output_len != 0 and output_len % agent_token == 0:
                    send_data[rid] == output_text
                    send_data[rid].finished= False
    send the send_data to B

======================================================
N2 
n2_need_act =  react_num // 2 

bs = args.bs

finished_seq = 0
reqs = []
for i in range(0, bs):
    reqs.append([str(i), init_prompt[i], REQ_TYPE.AGENT_PREFILL])
    reactB[str(i)] = 0
start = 0
while finished_seq != bs:
    if start == 0:
       start = 1
    else:
        data = recv_from_A
        for d in data:
            rid = d.rid
            finished = d.finished 
            output = d.output_text
            if finished: #do prefill + decode
                reqs.append([rid, prompt, output_text,PREFILL_DECODE])
            else:
                if agent_prefill_marked[rid] == True:#previous agent prefill is done
                    reqs.append([rid, prompt, output_text,AGENT_PREFILL])
                else:
                    waiting_reqs.append([rid, prompt, output_text,AGENT_PREFILL])
    while reqs:
        rid, prompt, req_type = reqs.pop()
        if req_type == PREFILL_DECODE:
            if counter[rid] < n2_need_react:
                add the requst into the engine ,use sampling_params   for prefill + decode 
        else:
            use agent_prefill__params   for agent_prefill
    
    request_output = engine.step()
    send_data = dict()
    for req in request_out:
        rid = req.rid
        output = req.output
        output_len = req.output_len
        req_type = req.type 
        if req.finished:
            if req_type  == PREFILL_DECODE:
                send_data[rid] = output,xxxx
                if counter[rid] == n2_need_act:
                    finished_seqs +=1 
  
            elif req_type == AGENT_PREFILL:
                agent_prefill_marked[rid]  = True
        else:
            if req_type  == PREFILL_DECODE:
                if output_len != 0 and output_len % agent_token == 0:
                    send_data[rid] = output,xxxx
    send the send_data into A