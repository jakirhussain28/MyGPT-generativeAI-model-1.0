import torch
import torch.nn.functional as F
from model import GPT, GPTConfig

# === Load Data ===
with open("testing/conversations_5k.txt", "r", encoding="utf-8") as f:
    text = f.read()

# === Tokenizer ===
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# === Hyperparameters ===
block_size = 128
batch_size = 32
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# === Model ===
config = GPTConfig()
config.vocab_size = vocab_size
config.block_size = block_size
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# === Training Loop ===
for step in range(max_iters):
    model.train()
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            _, val_loss = model(*get_batch("val"))
        print(f"Step {step}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")

# === Save Model and Generate Sample ===
torch.save(model.state_dict(), "my-gpt.pth")

context = torch.tensor([stoi['H']], dtype=torch.long).unsqueeze(0).to(device)
print("Generated:\n", decode(model.generate(context, max_new_tokens=300)[0].tolist()))
