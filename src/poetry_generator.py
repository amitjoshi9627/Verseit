import engine


def get_poem(seed_text):
    generated_poem = engine.predict(seed_text)
    return generated_poem
