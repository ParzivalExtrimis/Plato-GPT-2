import os
import torch
from model import GPTLanguageModel, Config
from web_handler import Web_handler
import random

run_r = random.randint(1000, 3000)
start = '\n' # can be anything ( shorter than block size, to seed the generation)
n_samples = 2
output_len = 5000

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

handle = Web_handler('')
checkpoints_dir = os.path.join('checkpoints')
handle.download_from_datastore('checkpoint', checkpoints_dir)
checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pt'), map_location=device)

#load model with meta at checkpoint
conf = Config(**checkpoint['model_args'])
model = GPTLanguageModel(conf)
state_dict = checkpoint['model']

# clean up
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

#get loaded model and set model to eval mode
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# get encodings
stoi = checkpoint['encodings']['stoi']
itos = checkpoint['encodings']['itos']

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l]) 

start_ids = encode(start)
idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

os.makedirs(os.path.join('samples'), exist_ok=True)
#generate
with open(os.path.normpath(os.path.join('samples', f'{run_r}.txt')), 'w', encoding='latin-1') as f:
    f.write('Title: Using lower learning rate and higher decay weights. Val loss decline has been observed to be smoother.')
    f.write(f'Parameters: {checkpoint["model_parameters"]}\n')
    f.write(f'Hyperparams: {checkpoint["model_args"]}\n')
    f.write(f'Val loss: {checkpoint["best_val_loss"]}\n')
    f.write(f'Trained at: {checkpoint["timestamp"]}\n')
    
    for _ in range(n_samples):
        out = model.generate(idx, output_len).squeeze(0)
        f.write(decode(out.tolist()))
        f.write('\n----------------------\n')


