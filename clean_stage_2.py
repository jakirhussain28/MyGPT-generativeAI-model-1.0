import re

def clean_whatsapp_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty or deleted messages
        if not line or "This message was deleted" in line:
            continue

        # Remove URLs
        line = re.sub(r'https?://\S+', '', line)

        # Remove phone numbers
        line = re.sub(r'\+?\d[\d -]{8,}\d', '', line)

        # Remove email addresses
        line = re.sub(r'\S+@\S+', '', line)

        # Remove special characters (keep basic punctuation)
        line = re.sub(r'[^\w\s.,?!\'"@#&()-]', '', line)

        # Normalize whitespace
        line = re.sub(r'\s+', ' ', line).strip()

        if line:
            cleaned_lines.append(line)

    return cleaned_lines


# Save cleaned text
cleaned_chat = clean_whatsapp_text('cleaned_chat.txt')

with open('cleaned_input_for_generator.txt', 'w', encoding='utf-8') as f:
    for line in cleaned_chat:
        f.write(line + '\n')

print("✅ Chat cleaned and saved to cleaned_output.txt")
