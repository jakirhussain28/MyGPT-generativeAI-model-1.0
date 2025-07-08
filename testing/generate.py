import torch
from model import GPT, GPTConfig

def load_model(vocab_size):
    config = GPTConfig()
    config.vocab_size = vocab_size
    model = GPT(config)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

def main():
    # Load token mappings
    with open("cleaned_input_for_generator.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    def encode(s): return [stoi.get(c, 0) for c in s]
    def decode(l): return ''.join([itos[i] for i in l])
    
    # Load model
    model = load_model(len(chars))
    
    # Generation loop
    print("Engineering Dialogue GPT - Type 'exit' to quit")
    while True:
        user_input = input("\n💬 You: ").strip()
        if user_input.lower() in {'exit', 'quit'}:
            break
            
        if not user_input:
            continue
            
        # Format prompt
        prompt = f"User 1: {user_input}\nUser 2:"
        context = torch.tensor([encode(prompt)], dtype=torch.long)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                context,
                max_new_tokens=200,
                temperature=0.7,
                top_k=50
            )[0].tolist()
        
        # Decode and display
        full_text = decode(output)
        response = full_text.split('User 2:')[-1].strip()
        print(f"\nAI Lab Partner:\n{response}")

if __name__ == "__main__":
    main()