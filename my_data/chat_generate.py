import random

topics = ["sports", "movies", "food", "travel", "tech", "work", "weather", "music"]
questions = ["What do you think about {}?", "How do you like {}?", "Ever tried {}?", "Been to {} lately?"]
responses = ["I love {}!", "Not a fan of {}.", "{} is okay.", "{} changed my life!"]

with open("conversations_5k.txt", "w") as f:
    for _ in range(5000):  # Adjust for line count
        topic = random.choice(topics)
        f.write(f"User 1: {random.choice(questions).format(topic)}\n")
        f.write(f"User 2: {random.choice(responses).format(topic)}\n")
        f.write(f"User 1: {random.choice(['Thanks!', 'Cool!', 'Interesting...'])}\n\n")