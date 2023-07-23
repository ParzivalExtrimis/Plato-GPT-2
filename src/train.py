import datetime
import os
import pytz
import torch
import time
import json
import math
import subprocess
import argparse
import mlflow
import mlflow.pytorch
from azureml.core import Run
from model import GPTLanguageModel, Config
from web_handler import Web_handler

def format_time(secs: float) -> str:
    return time.strftime('%H:%M:%S', time.gmtime(secs))
    
def main(args):

    torch.manual_seed(1337)
    # Start Logging
    run = Run.get_context()
    run_id = run._run_id
    if isinstance(run_id, str):
        mlflow.start_run(run_id=run_id)
    else:
        mlflow.start_run()

    mlflow.autolog()

    #state vars
    epoch = 0

    # hyperparameters
    start_fresh = args.start_fresh
    always_override_checkpoint = args.always_override_checkpoint

    batch_size = args.batch_size 
    block_size = args.block_size
    max_epoch = args.max_epoch
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    eval_iters = args.eval_iters
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    bias = args.bias

    use_decay = args.use_decay
    weight_decay = args.weight_decay
    beta1 = args.beta1
    beta2 = args.beta2
    learning_rate = args.learning_rate
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.lr_decay_iters
    min_lr = args.min_lr

    config_f = args.config
    # ------------

    w_handle = Web_handler(config_f)
    input_path = args.data

    # to log the dataset being used
    dataset = {
        'name' : 'Sam-Harris-Podcast-Transcripts',
        'path' : input_path
    }

    # log params
    params = {
            'start_fresh' : start_fresh,
            'batch_size' : batch_size, 
            'block_size' : block_size,
            'max_epcoh' : max_epoch,
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
            'use_decay' : use_decay,
            'weight_decay' : weight_decay,
            'beta1' : beta1,
            'beta2' : beta2,
            'learning_rate' : learning_rate,
            'warmup_iters' : warmup_iters,
            'lr_decay_iters' : lr_decay_iters,
            'min_lr' : min_lr,
    }

    mlflow.log_params(params)

    with open(input_path, 'r', encoding='latin-1') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    dataset['size'] = len(text)
    dataset['vocab_size'] = vocab_size

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
    
    # make checkpoints dir for instance; at fresh use for saving checkpoints;
    #                                    at resume use for downloading and loading checkpoints as well as saving
    checkpoints_dir = os.path.join('checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # set parameter options to be set for  each train instance
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                        bias=bias, vocab_size=None, dropout=dropout)

    #starting from scratch
    if start_fresh:
        print('Starting model training from scratch...')
        model_args['vocab_size'] = vocab_size

        conf = Config(**model_args)
        model = GPTLanguageModel(conf)
        m = model.to(device)
        best_val_loss = 1e9

    else:
        # download saved instances from DataStore
        w_handle.download_from_datastore('checkpoint', checkpoints_dir)
        checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pt'), map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            if model_args[k] != checkpoint_model_args[k]:
                print(f'New parameters fount. Does not match checkpoint. Run mode set to [Start_fresh = {start_fresh}]. Overwriting parameters - {k}, Old value: {model_args[k]} - New value: {checkpoint_model_args[k]}')
                mlflow.log_param(f'new-{k}', checkpoint_model_args[k])
            model_args[k] = checkpoint_model_args[k]

        # create the model
        conf = Config(**model_args)
        model = GPTLanguageModel(conf)
        state_dict = checkpoint['model']
        
        # anomalous prefix in state_dict, removed.
        unwanted_prefix = '_orig_mod.'
        for k,_ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f'Resuming model training from checkpoint [saved at: {checkpoint["timestamp"]}]')

    if not always_override_checkpoint:
        # download saved instances from DataStore
        try:
            w_handle.download_from_datastore('checkpoint', checkpoints_dir)
            checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pt'), map_location=device)
            best_val_loss = checkpoint['best_val_loss']
        except Exception as e:
            print(e, '\n\n','Could not find checkpoint at specified location at the Datastore. Continuing with fresh best loss...')
        finally:
            pass

    #decayed learning rate
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    #=========================================================================================================

    # create a PyTorch optimizer
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    if not start_fresh:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    start = time.time()

    last_train_loss = 1e9
    last_val_loss = 1e9

    while True:
        
        lr = get_lr(epoch) if use_decay else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # every once in a while evaluate the loss on train and val sets
        if epoch % eval_interval == 0:
            losses = estimate_loss()
            last_train_loss = losses['train'].item()
            last_val_loss = losses['val'].item()
            mlflow.log_metric('Training Loss', losses['train'])
            mlflow.log_metric('Validation Loss', losses['val'])

            print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if epoch % save_interval == 0 and epoch > 0:
            #track gpu stats
            gpu_stats = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            curr_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
            print(f'GPU Stats: {curr_time}\n {gpu_stats}')
            mlflow.log_text(gpu_stats, os.path.join('gpu_stats', f'{curr_time}.txt'))
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss # set new record, save state 
                checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'epoch': epoch,
                            'model_parameters' : model.get_num_params(),
                            'best_val_loss': best_val_loss,
                            'timestamp' : curr_time,
                            'encodings': {
                                'stoi' : stoi,
                                'itos' : itos,
                            },
                }
                print(f"saving checkpoint to {os.path.abspath(checkpoints_dir)}")
                torch.save(checkpoint, os.path.join(checkpoints_dir, 'checkpoint.pt'))
                w_handle.upload_to_datastore(
                filepath = os.path.join(checkpoints_dir, 'checkpoint.pt'),
                name = 'checkpoint',
                description = 'Checkpoint for torch.save() last commit.',
                )
                checkpoint = None # free up memory
        
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        #backward pass
        loss.backward()
        # step
        optimizer.step()

        epoch += 1
        if(epoch > max_epoch):
            break

    end = time.time()

    #==========================================================================================================

    #ouput logs directory
    logs_dir = os.path.join('outputs', str(run_id))
    os.makedirs(logs_dir, exist_ok=True)

    # generate from the model; and logging
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    with open(os.path.join(logs_dir, 'out.txt'), 'w') as out_f:
        out_f.write(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
    w_handle.upload_to_datastore(
        filepath = os.path.join(logs_dir, 'out.txt'),
        name = 'model_generated_output',
        description = 'Text sample generated by model after training',
    )

    elapse_interval_sec = end - start
    elapsed_time = format_time(elapse_interval_sec)

    snapshot = {
        'params' : params,
        'dataset' : dataset,
        'training_time': elapsed_time,
        'last_train_loss' : last_train_loss,
        'last_val_loss' : last_val_loss,
        'best_val_loss' : best_val_loss,
    }
        
    with open(os.path.join(logs_dir,'run_meta.json'), 'w') as fp:
        json.dump(snapshot, fp, indent=4)

    mlflow.log_artifact(os.path.join(logs_dir,'run_meta.json'))

    mlflow.end_run()
    run.complete()

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, help='Location of Dataset file or mount.')
    parser.add_argument('--config', type=str, help='Location of Config file or mount.')
    parser.add_argument('--start_fresh', type=bool, default=False, required=False, help='Flag indicates whether to use checkpoints to load at training.')
    parser.add_argument('--always_override_checkpoint', type=bool, default=False, required=False, help='Flag indicates whether to override checkpoints even when the current loss is higher than overall best at training.')

    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Number of parallel examples to be used per epoch.')
    parser.add_argument('--block_size', type=int, default=288, required=False, help='Context window size of the transformer.')
    parser.add_argument('--max_epoch', type=int, default=10000, required=False, help='Total number of iterations for training.')
    parser.add_argument('--eval_interval', type=int, default=250, required=False, help='Iterations to wait until next loss evaluation.')
    parser.add_argument('--save_interval', type=int, default=1000, required=False, help='Iterations to wait until next checkpoint save.')
    parser.add_argument('--use-cuda', type=bool, default=True, required=False, help='Flag indicates whether to use CUDA at training.')
    parser.add_argument('--eval_iters', type=int, default=200, required=False, help='Number of samples to use in-order to smooth out loss over batches.')
    parser.add_argument('--n_embd', type=int, default=768, required=False, help='Size of the embedding dimension.')
    parser.add_argument('--n_head', type=int, default=12, required=False, help='Number of attention heads.')
    parser.add_argument('--n_layer', type=int, default=12, required=False, help='Number of times to loop over tranformer layers.')
    parser.add_argument('--dropout', type=float, default=0.0, required=False, help='Dropout Ratio')
    parser.add_argument('--bias', type=bool, default=True, required=False, help='Flag indicates whether to use biases in Linear and LayerNorm layers.')

    # optimizer args
    parser.add_argument('--use_decay', type=bool, default=True, required=False, help='Flag indicated whether to use learning rate decay ( cosine decay ).')
    parser.add_argument('--learning_rate', type=float, default=6e-6, required=False, help='The magnitude at which the optimizer step changes the weights.')
    parser.add_argument('--weight_decay', type=float, default=6e-1, required=False, help='The magnitude at which the optimizer step changes the weights.')
    parser.add_argument('--beta1', type=float, default=0.9, required=False, help='Variable controls decay parameters.')
    parser.add_argument('--beta2', type=float, default=0.95, required=False, help='Variable controls decay parameters.')
    parser.add_argument('--warmup_iters', type=int, default=100, required=False, help='Initial iterations to run linear lr increment upto default lr.')
    parser.add_argument('--lr_decay_iters', type=int, default=7500, required=False, help='The amount of iterations upto which decay applies. Defaults to min_l after.')
    parser.add_argument('--min_lr', type=float, default=6e-7, required=False, help='The magnitude at which the optimizer step changes the weights.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)