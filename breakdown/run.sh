# num_acts=(2 3 4 5 6 7)
num_acts=(3 4)
#request_rates=(5 6 7 8 9 10 11 12 13 14 15)
request_rates=(6 7 8 9 11)

# num_acts=(2)
# request_rates=(2)
# num_system_prompts=(20 50 70 100)
# request_rates=(2 3 4 5 6 7 8 9 10)
num_acts=(3 4 5 6)
#request_rates=(5 6 7 8 9 10 11 12 13 14 15)
request_rates=(1 2 3 4 5 6)


# num_system_prompts=(100)
# request_rates=(30 35) #TODO (40, 45)

# request_rates=(40 45) #TODO (40, 45)

DURATION=3


for num_act in ${num_acts[@]}
do 
    for request_rate in ${request_rates[@]}
    do
        python3 breakdown_vllm.py --duration $DURATION --num-act $num_act  --model /data/Meta-Llama-3.1-8B-Instruct --model-name "llama3_1_8B" --request-rate $request_rate  > "./log/vllm_react_run_7b_num_act_${num_act}_request_rate_${request_rate}.log" 2>&1
 
        # git add .
        # git commit -m "finished run with num_system_prompt = $num_system_prompt and request_rate = $request_rate"

        # git push origin master 

        ps -aux | grep "breakdown_vllm.py" | grep -v grep | awk '{print $2}' | xargs kill -9
        # ps -aux | grep "run.sh" | grep -v grep | awk '{print $2}' | xargs kill -9
        # ps -aux | grep "bench_agent_system_prompt.py" | grep -v grep | awk '{print $2}' | xargs kill -9
        # ps -aux | grep "bench_vllm_system_prompt.py" | grep -v grep | awk '{print $2}' | xargs kill -9
        # ps -aux | grep "run_vllm_7b" | grep -v grep | awk '{print $2}' | xargs kill -9

    done
done 


# num_acts=(3 4 5 6)
# #request_rates=(5 6 7 8 9 10 11 12 13 14 15)
# request_rates=(4 5)


# # num_system_prompts=(100)
# # request_rates=(30 35) #TODO (40, 45)

# # request_rates=(40 45) #TODO (40, 45)

# DURATION=3


# for num_act in ${num_acts[@]}
# do 
#     for request_rate in ${request_rates[@]}
#     do
#         python3 breakdown_vllm.py --duration $DURATION --num-act $num_act  --model /data/Meta-Llama-3.1-8B-Instruct --model-name "llama3_1_8B" --request-rate $request_rate  > "./log/vllm_react_run_7b_num_act_${num_act}_request_rate_${request_rate}.log" 2>&1
 
#         # git add .
#         # git commit -m "finished run with num_system_prompt = $num_system_prompt and request_rate = $request_rate"

#         # git push origin master 

#         ps -aux | grep "breakdown_vllm.py" | grep -v grep | awk '{print $2}' | xargs kill -9
#         ps -aux | grep "run.sh" | grep -v grep | awk '{print $2}' | xargs kill -9
#         # ps -aux | grep "bench_agent_system_prompt.py" | grep -v grep | awk '{print $2}' | xargs kill -9
#         # ps -aux | grep "bench_vllm_system_prompt.py" | grep -v grep | awk '{print $2}' | xargs kill -9
#         # ps -aux | grep "run_vllm_7b" | grep -v grep | awk '{print $2}' | xargs kill -9

#     done
# done 