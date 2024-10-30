# React Workload

### react_num % 2 == 1(application r0->r1->r0...)
when react_num % 2 == 1, it means the r0 is the last request
 ```
not the last request
   
    output_text_len = len(request_output.outputs[0].token_ids)
    is_terminate = info[rx0].terminate_application()
    final_terminate = info[rx1].final_terminate()
    output_text = request_output.outputs[0].text 
    now = time.time()
    
    if current_react_num % 2 == 1:#ro prefill + decode, r1 agent prefill 
        #we must ensure r1 done, after r0 done, we can do agent prefill for next llm call
        r0 done:
            r1 not done: #r0 done before r1, store the r0's result
                #store the r0 into a dict 
                fin_req =  FinishReq(r0_id, r1_id, output_text, output_text_len,now)
                finished_req[r0_id] =  fin_req
                finished_req[r1_id] = fin_req
                #when r0_id is in the finished_req, it means the r0 is done 
            r1 done:  #r1 done before r0
                #finish r0 prefill + decode and finish r1 agent prefill
                #now start to r1 prefill+decode
                #calculate the duration and tokens
                if current_react_num != num_act:
                    reqs.append([rx1, info[rx1].r1_user_prompt + output_text])
                info[rx1].r2_user_prompt = info[rx1].r1_user_prompt + output_text
                info[rx1].total_duration += now - info[rx1].arr1
                info[rx1].total_token += len(request_output.outputs[0].token_ids)
                info[rx0].r2_react_num += 1

        r1 done:
            marked r1 as done
            if r0 in  finished_req: #r0 done before r1 
                #calculate the duration and tokens
                if current_react_num != num_act:
                    reqs.append([rx1, info[rx1].r1_user_prompt + output_text])
                info[rx1].r2_user_prompt = info[rx1].r1_user_prompt + output_text
                info[rx1].total_duration += now - info[rx1].arr1
                info[rx1].total_token += len(request_output.outputs[0].token_ids)
                info[rx0].r2_react_num += 1
                finished_reqs.pop(rx0)
                finished_reqs.pop(rx1)
    
    if current_react_num % 2 == 0:#r0 agent prefill, r1 prefill + decode
        #we must ensure r0 done, after r1 done, we can do agent prefill for next llm call
        r1 done:
            r0 not done:#r1 done before the r0
                #store the r1's result 
                fin_req =  FinishReq(r0_id, r1_id, output_text, output_text_len,now)
                finished_req[r0_id] =  fin_req
                finished_req[r1_id] = fin_req
                #when r1_id is in the finished_req, it means the r1 is done 
            r0 done: #r0 done
                #finish the r0 agent prefill

                if r1_id in finished_req:#r1 done before r0
                    if current_react_num != num_act:
                        reqs.append([rx1, info[rx1].r2_user_prompt + output_text])
                info[rx1].r1_user_prompt  = info[rx1].r2_user_prompt + output_text
                info[rx1].total_duration += now - info[rx0].arr2
                info[rx1].totals_token += len(request_output.outputs[0].token_ids)
                info[rx1].r1_react_num +=1
                finished_reqs.pop(rx0)
                finished_reqs.pop(rx1)
                
        r0 done:
            mark r0 as done
            if r1 in finishsed_reqs: #r1 done before r0 
                if r1_id in finished_req:#r1 done before r0
                    if current_react_num != num_act:
                        reqs.append([rx1, info[rx1].r2_user_prompt + output_text])
                    info[rx1].r1_user_prompt  = info[rx1].r2_user_prompt + output_text
                    info[rx1].total_duration += now - info[rx0].arr2
                    info[rx1].totals_token += len(request_output.outputs[0].token_ids)
                    info[rx1].r1_react_num +=1
                    finished_reqs.pop(rx0)
                    finished_reqs.pop(rx1)

if the last request:
    fin_req = FinishReq(rx0, rx1, output_text, output_text_len,now)
    finished_reqs[rx1] = fin_req
    finished_reqs[rx0] = fin_req 
    info[rx0].r1_react_num += 1 #TODO(xiao):this may have bug, bc it can only work when the num_act is even


 ```