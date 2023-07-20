import os
import torch
import random
import time
import json
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel, Config

# hyperparameters
start_fresh = True
batch_size = 32 
block_size = 128
max_epcoh = 5000
eval_interval = 500
save_interval = 1500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 250
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
bias = True
# ------------

torch.manual_seed(1337)
run_id = random.randint(100000, 200000)

input_path = 'data.txt'

# to log the dataset being used
dataset = {
    'name' : 'Sam-Harris-Podcast-Transcripts',
    'path' : os.path.normpath(os.path.abspath(input_path))
}

with open(input_path, 'r', encoding='latin-1') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# set parameter options to be set for  each train instance
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout)

#starting from scratch
if start_fresh is True:
    model_args['vocab_size'] = vocab_size

    conf = Config(**model_args)
    model = GPTLanguageModel(conf)
    m = model.to(device)
    best_val_loss = 1e9

checkpoints_dir = os.path.join('checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

#==========================================================================================================

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = time.time()

last_train_loss = 1e9
last_val_loss = 1e9

for epoch in range(max_epcoh):

    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == max_epcoh - 1:
        losses = estimate_loss()
        last_train_loss = losses['train'].item()
        last_val_loss = losses['val'].item()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if epoch % save_interval == 0 or epoch == max_epcoh -1:
        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss # set new record, save state 
            checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'encodings': {
                            'stoi' : stoi,
                            'itos' : itos,
                        },
            }
            print(f"saving checkpoint to {os.path.abspath(checkpoints_dir)}")

            torch.save(checkpoint, os.path.join(checkpoints_dir, 'checkpoint.pt'))
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end = time.time()

#==========================================================================================================

#ouput logs directory
logs_dir = os.path.join('outputs', str(run_id))
os.makedirs(logs_dir, exist_ok=True)

# generate from the model; and logging
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

with open(os.path.join(logs_dir, 'out.txt'), 'w') as out_f:
    out_f.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

elapse_interval_sec = end - start
elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapse_interval_sec))

params = {
        'start_fresh' : start_fresh,
        'batch_size' : batch_size, 
        'block_size' : block_size,
        'max_epcoh' : max_epcoh,
        'eval_interval' : eval_interval,
        'save_interval' : save_interval,
        'learning_rate' : learning_rate,
        'device' : device,
        'eval_iters' : eval_iters,
        'n_embd' : n_embd,
        'n_head' : n_head,
        'n_layer' : n_layer,
        'dropout' : dropout,
        'bias' : bias, 
}

snapshot = {
    'params' : params,
    'dataset' : dataset,
    'last_train_loss' : last_train_loss,
    'last_val_loss' : last_val_loss,
    'best_val_loss' : best_val_loss,
}
    
with open(os.path.join(logs_dir,'run_meta.json'), 'w') as fp:
    json.dump(snapshot, fp, indent=4)