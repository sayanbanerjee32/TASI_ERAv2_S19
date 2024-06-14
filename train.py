import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import pickle
import requests

from model import BigramLanguageModel, ModelConfig

out_dir = './saved_model/'
data_dir = './data/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters - GPU
if device == 'cuda':
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    n_embed = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
# ------------------- CPU
else:
    batch_size = 16
    block_size = 64
    max_iters = 10
    eval_interval = 5
    learning_rate = 1e-3
    eval_iters = 3
    n_embed = 8
    n_head = 4
    n_layer = 2
    dropout = 0.1

# prepare the dataset
torch.manual_seed(1337)
# tiny shakespeare dataset
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
# read the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique cahancters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from chars to ints
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

# lets split into trainh and test
n = int(0.9*len(data)) # 90% will be train, rest validation
train_data = data[:n]
val_data = data[n:]

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
os.makedirs(data_dir, exist_ok=True)
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, 
                    n_embed=n_embed, block_size=block_size,
                  vocab_size=vocab_size, dropout=dropout)
modconf = ModelConfig(**model_args)
model = BigramLanguageModel(modconf).to(device)
# m = model.to(device)

# optimiser
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
best_val_loss = 1e9
os.makedirs(out_dir, exist_ok=True)
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        if iter > 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_args': model_args
                }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
