import torch
from model import GPT, GPTConfig

# Load Data and Char Mappings
with open("testing/conversations_5k.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s): return [stoi.get(c, 0) for c in s]  # default to 0 for unknown
def decode(l): return ''.join([itos[i] for i in l])

# Load Model
config = GPTConfig()
config.vocab_size = vocab_size
config.block_size = 128
model = GPT(config)
model.load_state_dict(torch.load("my-gpt.pth", map_location='cpu'))
model.eval()

# User Prompt + Generation
while True:
    user_input = input("\n💬 Enter your message: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Exiting chat generation.")
        break

    if len(user_input) == 0:
        print("⚠️ Please enter something.")
        continue

    # Encode user input to token IDs
    context = torch.tensor([encode(user_input)], dtype=torch.long)

    # Generate continuation
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=200)[0].tolist()

    # Decode and print generated text
    full_text = decode(output)
    print("\nMyGPT:\n")
    print(full_text)
