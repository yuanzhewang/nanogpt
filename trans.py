import requests
import os
import torch
import random
import time
import math
import numpy as np

#####################################
# Hyper-parameters
#####################################

seed = 971

training_data_ratio = 0.9
block_size = 8
batch_size = 32

num_layers = 3
num_heads = 4
num_embed = 32
dropout = 0.2

training_steps = 10000
learning_rate = 0.001

eval_interval = 1000
eval_examples = 100

inference_num = 5
inference_length = 100

#####################################
# Set Random Seed for Reproducibility
#####################################

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f'Set all random seeds to {seed}')

#####################################
# Device
#####################################

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# Trick: for small examples, CPU is faster
device = 'cpu'
print(f'Code runs on {device}')

#####################################
# Data Downloading And Reading
#####################################

# 1. Download the dataset
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
file_path = 'input.txt'

if not os.path.exists(file_path):
    print("Downloading Tiny Shakespeare...")
    data = requests.get(url).text
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)
else:
    print("Dataset already downloaded.")

# 2. Import (Read) the data
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 3. Inspect the data
print(f"Length of dataset in characters: {len(text)}")

#####################################
# Tokenizer
#####################################

vocab = sorted(set(text))
vocab_size = len(vocab)
print('Vocabulary Size is: ', vocab_size)

ctoi = {c : i for i, c in enumerate(vocab)}
itoc = {i : c for i, c in enumerate(vocab)}

encoder = lambda s : [ctoi[c] for c in s]
decoder = lambda a : ''.join([itoc[i] for i in a])

full_data = torch.tensor(encoder(text), dtype=torch.long)

#####################################
# Chunk Data
#####################################

training_data_size = int(training_data_ratio * len(full_data))
training_data = full_data[:training_data_size]
validation_data = full_data[training_data_size:]
print('Training data size: ', len(training_data))
print('Validation data size: ', len(validation_data))

def get_batch(split):
    data = training_data if split == 'training' else validation_data
    start = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i : i + block_size] for i in start])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start])
    return x.to(device), y.to(device)

#####################################
# Bigram Model
#####################################

class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.embedding(x)
        loss = None
        B, T, C = logits.shape
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

#####################################
# Transformer Model
#####################################

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, block_size, num_embed, num_heads, dropout) -> None:
        super().__init__()
        assert(num_embed % num_heads == 0)
        self.num_heads = num_heads
        self.head_size = num_embed // num_heads
        self.Wk = torch.nn.Linear(num_embed, num_embed, bias=False)
        self.Wv = torch.nn.Linear(num_embed, num_embed, bias=False)
        self.Wq = torch.nn.Linear(num_embed, num_embed, bias=False)
        self.Wo = torch.nn.Linear(num_embed, num_embed, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.AttnDropout = torch.nn.Dropout(p=dropout)
        self.OutputDropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, x):
        # Parameters
        B, T, C = x.shape
        N = self.num_heads
        H = self.head_size
        K = self.Wk(x)
        V = self.Wv(x)
        Q = self.Wq(x)

        # Input projection
        K = K.view(B, T, N, H).permute(0,2,1,3)
        V = V.view(B, T, N, H).permute(0,2,1,3)
        Q = Q.view(B, T, N, H).permute(0,2,1,3)

        # Attention matrix
        Attn = Q @ K.transpose(-2, -1)
        Attn = Attn / math.sqrt(H)

        # Casual Masking
        Attn = Attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax
        Attn = torch.nn.functional.softmax(Attn, dim=-1)

        # Attention dropout
        Attn = self.AttnDropout(Attn)

        # Apply attention
        x = Attn @ V
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # Output projection
        x = self.Wo(x)

        # Output dropout
        x = self.OutputDropout(x)
        return x

class FeedForward(torch.nn.Module):
    def __init__(self, num_embed, dropout) -> None:
        super().__init__()
        self.W1 = torch.nn.Linear(num_embed, num_embed * 4, bias=True)
        self.GELU = torch.nn.GELU()
        self.W2 = torch.nn.Linear(num_embed * 4, num_embed, bias=True)
        self.Dropout = torch.nn.Dropout(p = dropout)

    def forward(self, x):
        x = self.W1(x)
        x = self.GELU(x)
        x = self.W2(x)
        x = self.Dropout(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, block_size, num_embed, num_heads, dropout) -> None:
        super().__init__()
        self.LN1 = torch.nn.LayerNorm(num_embed)
        self.LN2 = torch.nn.LayerNorm(num_embed)
        self.Attn = CausalSelfAttention(block_size, num_embed, num_heads, dropout)
        self.FF = FeedForward(num_embed, dropout)

    def forward(self, x):
        x = x + self.Attn(self.LN1(x))
        x = x + self.FF(self.LN2(x))
        return x


class TransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, block_size, num_embed, num_heads, num_layers, dropout) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, num_embed)
        self.pos_embedding = torch.nn.Embedding(block_size, num_embed)
        self.blocks = torch.nn.Sequential(*[Block(block_size, num_embed, num_heads, dropout) for i in range(num_layers)])
        self.Dropout = torch.nn.Dropout(p=dropout)
        self.LNf = torch.nn.LayerNorm(num_embed)
        self.Linf = torch.nn.Linear(num_embed, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.embedding(x) + self.pos_embedding(torch.arange(0, T, dtype=torch.long, device=x.device))
        x = self.Dropout(x)
        x = self.blocks(x)
        x = self.LNf(x)
        logits = self.Linf(x)
        loss = None
        
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

#####################################
# Training Loop
#####################################

# model = BigramModel(vocab_size=vocab_size)
model = TransformerModel(vocab_size, block_size, num_embed, num_heads, num_layers, dropout)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

time_start = time.time()

def validate():
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for _ in range(eval_examples):
            x, y = get_batch('validation')
            _, loss = model(x, y)
            loss_sum += loss.item()
    model.train()
    return loss_sum / eval_examples

for s in range(training_steps):
    # 1. Get data
    x, y = get_batch('training')
    # 2. Forward
    logits, loss = model(x, y)
    # 3. Validation
    if s % eval_interval == 0 or s == training_steps - 1:
        validation_loss = validate()
        time_spent = time.time() - time_start
        print(f'Step: {s} | Time: {time_spent:0.3f} | Training loss: {loss.item():0.5f} | Validation loss: {validation_loss:0.5f}')

    # 4. Reset gradients
    optimizer.zero_grad()
    # 5. Backward
    loss.backward()
    # 6. Update parameters
    optimizer.step()

#####################################
# Inference Loop
#####################################

idx = torch.zeros((inference_num, 1), dtype=torch.long, device=device)

# Model to evaluation mode.
model.eval()

with torch.no_grad():
    for step in range(inference_length):
        # 1. Generate
        logits, _ = model(idx[:, -block_size:])
        # 2. Logits -> Probs
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        # 3. Sample
        new_id = torch.multinomial(probs, num_samples=1)
        # 4. Concatenate 
        idx = torch.cat((idx, new_id), dim=1)

for i_example, example in enumerate(idx):
    gen_text = decoder(example.tolist())
    print(f'Generation [{i_example}]: {gen_text}\n\n')

# In case we need to train model later on in the script.
model.train()