import requests
import os
import torch

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

full_data = encoder(text)

#####################################
# Chunk Data
#####################################

block_size = 8
batch_size = 4

training_data_size = int(0.9 * len(full_data))
training_data = full_data[:training_data_size]
validation_data = full_data[training_data_size:]
print('Training data size: ', len(training_data))
print('Validation data size: ', len(validation_data))

def get_batch(split):
    data = training_data if split == 'training' else validation_data
    

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

model = BigramModel(vocab_size=vocab_size)
x = torch.tensor([[5, 3, 21], [4, 2, 11]])
t = torch.tensor([[4, 5, 1], [5, 61, 1]])
y, loss = model(x, t)
print(x)
print(y)
print(loss)
print(x.shape)
print(y.shape)
print(loss.shape)