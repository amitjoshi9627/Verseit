import streamlit as st
from poetry_generator import get_poem

st.title("Poetry Generator")
seed_text = st.text_input("Enter Your Seed Text..")

if seed_text is not None:
    result = get_poem(seed_text)
    if st.button("Check Result"):
        st.write(result)
