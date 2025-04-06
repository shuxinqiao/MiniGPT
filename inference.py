import os
import time
import math
import pickle
import yaml
import argparse
from contextlib import nullcontext
import tiktoken

import numpy as np
import torch

from model import GPT, GPTConfig


if __name__ == "__main__":
    # Parse commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config file path')
    parser.add_argument('--weight', type=str, default=None, help='weight file path')
    parser.add_argument('--start', type=str, default="<|endoftext|>", help='text for completion')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples generate for single completion')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='context window limit')
    parser.add_argument('--temperature', type=float, default=1.0, help='< 1.0 means less random, > 1.0 means more random')
    parser.add_argument('--top_k', type=int, default=None, help='first k most likely tokens, will ignore others')
    args = parser.parse_args()

    # force config and weight path provided
    if args.config is None:
        raise ValueError('Please provide a config file path')
    if args.weight is None:
        raise ValueError('Please provide a weight file path')
    
    if args.start != "<|endoftext|>":
        start = args.start
    else:
        start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        

    # Load config YAML file
    with open(f"{args.config}", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load for security
    

    # Settings
    seed = config["training"]["seed"]
    device = config["system"]["device"]
    device_type = config["system"]["device"]
    dtype = config["system"]["dtype"]
        
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init from a model saved in a specific directory
    ckpt_path = os.path.join(args.weight)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    
    model.eval()
    model.to(device)

    # Tiktoken with gpt2
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
                print(decode(y[0].tolist()))
                print('---------------')