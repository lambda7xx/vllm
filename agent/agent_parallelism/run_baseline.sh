bs=256
num_act=5
# for b in ${bs[@]}
# do

#     python3 agent1.py --num-act $num_act --batch-size $b  > ./log/agent1_num_act_${num_act}_bs_${b}_chunk_prefill1024_prompt500.log 2>&1 &
#     python3 agent2.py --num-act $num_act --batch-size $b  > ./log/agent2_num_act_${num_act}_bs_${b}_chunk_prefill1024_prompt500.log 2>&1 &
#     #python3 baseline.py --num-act $num_act --batch-size $b  > ./log/baseline_num_act_${num_act}_bs_${b}_chunk_prefill1024_prompt500.log 2>&1 
#     #ps -aux | grep "agent" | grep -v grep | awk '{print $2}' | xargs kill -9


# done 

python3 baseline_agent1.py  --num-act $num_act --batch-size $bs  > ./log/agent1_num_act_${num_act}_bs_${bs}_chunk_prefill1024_prompt500.log 2>&1 &

python3 baseline_agent2.py  --num-act $num_act --batch-size $bs  > ./log/agent2_num_act_${num_act}_bs_${bs}_chunk_prefill1024_prompt500.log 2>&1 &

# ps -aux | grep "run" | grep -v grep | awk '{print $2}' | xargs kill -9

