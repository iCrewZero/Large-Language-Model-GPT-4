import numpy as np
import json
from tokenizer.tokenizer import Tokenizer
from tqdm import tqdm

tok = Tokenizer("tokenizer/tokenizer.model")

def encode_chat(example):
    ids = []
    for msg in example["messages"]:
        role = msg["role"]
        if role == "system": ids.append(tok.encode("<|system|>")[0])
        if role == "user": ids.append(tok.encode("<|user|>")[0])
        if role == "assistant": ids.append(tok.encode("<|assistant|>")[0])
        ids += tok.encode(msg["content"])
        ids.append(tok.encode("</s>")[0])
    return ids

data = []
with open("data/raw/chat.jsonl") as f:
    for line in tqdm(f):
        ex = json.loads(line)
        data.extend(encode_chat(ex))

arr = np.array(data, dtype=np.uint16)
arr.tofile("data/train.bin")
