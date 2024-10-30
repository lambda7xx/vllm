

bs=(2 4 8 16 32 64)
# bs=(2)

# #without agent parallelism
# for b in ${bs[@]}
# do
bs=(64)
#     python3 agent.py --num-act 3 --batch-size $b  > wo_agent_num_act3_bs${b}_chunk_prefill1024_prompt500.log 2>&1
# done 

#with agent parallelism
for b in ${bs[@]}
do

    python3 agent.py --num-act 3 --batch-size $b --agent-parallelism  > ./log/new_agent_num_act3_bs${b}_chunk_prefill1024.log 2>&1
done

# for b in ${bs[@]}
# do

#     python3 agent.py --num-act 3 --batch-size $b --agent-parallelism  > agent_num_act3_bs${b}_chunk_prefill1536_prompt1000.log 2>&1
# done
# python3 agent.py --num-act 3 --batch-size 2 --agent-parallelism  > agent_num_act3.log 2>&1

#/home/xiaoxiang/data/Llama-2-13b-hf 


# python3 agent.py --num-act 3 --batch-size 2 --agent-parallelism  --model /home/xiaoxiang/data/Llama-2-13b-hf  > agent_num_act3.log 2>&1

#ps -aux | grep "run" | grep -v grep | awk '{print $2}' | xargs kill -9

#ps -aux | grep "agent" | grep -v grep | awk '{print $2}' | xargs kill -9