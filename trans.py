from uu import encode
import requests
import os

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
print('Vocabulary Size is: ', len(vocab))

ctoi = {c : i for i, c in enumerate(vocab)}
itoc = {i : c for i, c in enumerate(vocab)}

encoder = lambda s : [ctoi[c] for c in s]
decoder = lambda a : ''.join([itoc[i] for i in a])