def clean_whatsapp_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        if '-->' not in line and len(line.strip()) > 0:
            if "<Media omitted>" in line:
                continue  # Skip media omitted lines
            # Remove timestamp, usernames, and system messages
            parts = line.split(': ')
            if len(parts) > 1:
                cleaned_lines.append(parts[1].strip())

    return '\n'.join(cleaned_lines)

cleaned_text = clean_whatsapp_chat("data/dataset/WhatsAppChat_Alumni_2024-25.txt")

with open("data/dataset/Alumni_2024-25_cleaned_chat.txt", "w") as f:
    f.write(cleaned_text)
