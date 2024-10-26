import json
from typing import Any, Dict, List, Optional, Tuple, Union

from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
import tiktoken

def _num_token_from_text(text: str, model: str = "gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text, allowed_special={'<|endoftext|>'}))


def _num_token_from_messages(messages: Union[List, Dict], model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.

    Retrieved from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb/
    """
    if isinstance(messages, dict):
        messages = [messages]

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if "gpt-4" in model or "gpt-3.5" in model:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""_num_token_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue

            # function calls
            if not isinstance(value, str):
                try:
                    value = json.dumps(value)
                except TypeError:
                    continue
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_tokens(input: Union[str, List, Dict], model: str = "gpt-4") -> int:
    """Count number of tokens used by an OpenAI model.
    Args:
        input: (str, list, dict): Input to the model.
        model: (str): Model name.

    Returns:
        int: Number of tokens from the input.
    """
    if isinstance(input, str):
        return _num_token_from_text(input, model=model)
    elif isinstance(input, list) or isinstance(input, dict):
        return _num_token_from_messages(input, model=model)
    else:
        raise ValueError(f"Input must be str, list or dict, but received {type(input)}")


path ="/home/xiaoxias/vllm_profile/ShareGPT_V3_unfiltered_cleaned_split.json"


def read_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {path}")
    #print(f"data[0]: {type(data[0])}")
    # print(len(data[0]["conversations"]))
    # print(data[0]["conversations"][0]["value"])
    # for key, value in data[0].items():
    #     print(f"key: {key}, value: {value}")
    prompt_300 = [] #<=300 
    prompt_400 = [] #300~500
    prompt_500 = [] # 500~600
    prompt_1000 = [] # 1000~2000
    prompt_2000 = [] # 2000~2000
    for d in data:
        d_len = len(d["conversations"])
        if d_len == 0 :
            continue
        print(f"len of conversations: {len(d['conversations'])}")
        value = d["conversations"][0]["value"]
        if count_tokens(value) <= 300:
            prompt_300.append(value)
        elif count_tokens(value) <= 400:
            prompt_400.append(value)
        elif count_tokens(value) >=500 and count_tokens(value) <= 600:
            prompt_500.append(value)
        elif count_tokens(value) >=1000 and count_tokens(value) <= 1200:
            prompt_1000.append(value)
        elif count_tokens(value) >=2000 and count_tokens(value) <= 2200:
            prompt_2000.append(value)
    #store the prompts into different files
    with open("prompt_300.json", "w") as f:
        json.dump(prompt_300, f)
    with open("prompt_400.json", "w") as f:
        json.dump(prompt_400, f)
    with open("prompt_500.json", "w") as f:
        json.dump(prompt_500, f)
    with open("prompt_1000.json", "w") as f:
        json.dump(prompt_1000, f)
    with open("prompt_2000.json", "w") as f:
        json.dump(prompt_2000, f)

read_json_file(path)