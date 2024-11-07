bs=8
num_act=3
# for b in ${bs[@]}
# do

#     python3 agent1.py --num-act $num_act --batch-size $b  > ./log/agent1_num_act_${num_act}_bs_${b}_chunk_prefill1024_prompt500.log 2>&1 &
#     python3 agent2.py --num-act $num_act --batch-size $b  > ./log/agent2_num_act_${num_act}_bs_${b}_chunk_prefill1024_prompt500.log 2>&1 &
#     #python3 baseline.py --num-act $num_act --batch-size $b  > ./log/baseline_num_act_${num_act}_bs_${b}_chunk_prefill1024_prompt500.log 2>&1 
#     # ps -aux | grep "agent" | grep -v grep | awk '{print $2}' | xargs kill -9


# done 

ps -aux | grep "agent" | grep -v grep | awk '{print $2}' | xargs kill -9

python3 agent1.py --num-act $num_act --batch-size $bs  > ./log/our_agent1_num_act_${num_act}_bs_${bs}_chunk_prefill1024_prompt500.log 2>&1 &

python3 agent2.py --num-act $num_act --batch-size $bs  > ./log/our_agent2_num_act_${num_act}_bs_${bs}_chunk_prefill1024_prompt500.log 2>&1 &

# ps -aux | grep "run" | grep -v grep | awk '{print $2}' | xargs kill -9

