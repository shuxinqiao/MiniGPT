import os
import time
import math
import pickle
import yaml
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig


# Data loader
def get_batch(data_dir, split, batch_size, context_length, device_type):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_length]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device_type, non_blocking=True), y.pin_memory().to(device_type, non_blocking=True)
    else:
        x, y = x.to(device_type), y.to(device_type)
    return x, y

# optimizer settings
def configure_optimizers(model, optimizer, weight_decay, learning_rate, betas, device_type, fused):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    if optimizer == "AdamW":
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=fused)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, fused=fused)

    return optimizer


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, ctx, data_dir, batch_size, context_length, device_type):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, split, batch_size, context_length, device_type)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    

if __name__ == "__main__":
    # Parse commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config file path')
    args = parser.parse_args()

    # force config path provided
    if args.config is None:
        raise ValueError('Please provide a config file path')
    

    # Load config YAML file
    with open(f"{args.config}", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load for security


    # Create output directory
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(config["training"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True    # Enable TensorFloat-32 for CUDA
    torch.backends.cudnn.allow_tf32 = True          # Enable TensorFloat-32 for cuDNN

    device_type = config["system"]["device"]

    
    # Model initialization
    context_length = config["data"]["context_length"]
    vocab_size = config["data"]["vocab_size"]
    n_layer = config["model"]["n_layer"]
    n_head = config["model"]["n_head"]
    embedding_dim = config["model"]["embedding_dim"]
    dropout = config["model"]["dropout"]
    bias = config["model"]["bias"]

    model_args = dict(context_length=1024, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                      embedding_dim=embedding_dim, dropout=dropout, bias=bias)
    
    # init training loop, can override if init_from='resume' (i.e. from a checkpoint)
    init_from = config["training"]["init_from"]
    iter_num = config["training"]["iter_num"] if init_from == 'resume' else 0
    best_val_loss = 1e9

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {output_dir}")
        ckpt_path = os.path.join(output_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device_type)
        checkpoint_model_args = checkpoint['model_args']

        for key in ['context_length', 'vocab_size', 'n_layer', 'n_head', 'embedding_dim', 'bias']:
            model_args[key] = checkpoint_model_args[key]

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']

        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    model = model.to(device_type)

    # optimizer
    optimizer = config["optimizer"]["optimizer"]
    weight_decay = config["optimizer"]["weight_decay"]
    learning_rate = config["optimizer"]["learning_rate"]
    betas = (config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    fused = config["optimizer"]["fused"]
    optimizer = configure_optimizers(model, optimizer, weight_decay, learning_rate, betas, device_type, fused)

    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None


    # compile the model
    if config["training"]["compile"]:
        print("compiling the model... (takes some time)")
        unoptimized_model = model
        model = torch.compile(model)


    # logging
    wandb_log = config["wandb"]["wandb_log"]
    if wandb_log:
        import wandb
        wandb.init(project=config["wandb"]["wandb_project"], name=config["wandb"]["wandb_run_name"], config=config)


    # training configs
    eval_interval = config["training"]["eval_interval"]
    eval_iters = config["training"]["eval_iters"]
    log_interval = config["training"]["log_interval"]
    max_iters = config["training"]["max_iters"]

    decay_lr = config["scheduler"]["decay_lr"]
    warmup_iters = config["scheduler"]["warmup_iters"]
    lr_decay_iters = config["scheduler"]["lr_decay_iters"]
    min_lr = config["scheduler"]["min_lr"]


    dtype = config["system"]["dtype"]
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))    # initialize a GradScaler. If enabled=False scaler is a no-op
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)

    grad_clip = config["optimizer"]["grad_clip"]

    # Data loader
    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]

    # training loop
    X, Y = get_batch(data_dir, "train", batch_size, context_length, device_type) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model # unwrap DDP container if needed
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, ctx, data_dir, batch_size, context_length, device_type)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or config["training"]["always_save_checkpoint"]:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {output_dir}")
                    torch.save(checkpoint, os.path.join(output_dir, 'ckpt.pt'))

        if iter_num == 0 and config["training"]["eval_only"]:
            break
            
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(data_dir, "train", batch_size, context_length, device_type)

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = 0#raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
