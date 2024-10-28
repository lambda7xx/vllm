

# python3 agent.py --num-act 3 --batch-size 1 > wo_agent_num_act3.log 2>&1

python3 agent.py --num-act 3 --batch-size 1 --agent-parallelism  > agent_num_act3.log 2>&1

#/home/xiaoxiang/data/Llama-2-13b-hf 


# python3 agent.py --num-act 3 --batch-size 2 --agent-parallelism  --model /home/xiaoxiang/data/Llama-2-13b-hf  > agent_num_act3.log 2>&1

#ps -aux | grep "run" | grep -v grep | awk '{print $2}' | xargs kill -9

#ps -aux | grep "agent" | grep -v grep | awk '{print $2}' | xargs kill -9