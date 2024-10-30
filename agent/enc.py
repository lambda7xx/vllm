from transformers import AutoTokenizer


model_name ="/home/xiaoxiang/data/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载tokenizer


# 输入文本
prompt = "Hello, how are you?"

# 编码文本
encoded_prompt = tokenizer.encode(prompt)

# 输出编码结果
print("Encoded Prompt:", encoded_prompt)
