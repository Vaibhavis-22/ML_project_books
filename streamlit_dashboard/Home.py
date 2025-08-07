import streamlit as st

st.set_page_config(page_title="Book Dashboard", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Roboto:wght@400;500&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            color: #333333;
        }

        h1 {
            font-family: 'Playfair Display', serif;
            font-size: 40px;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 40px;
        }

        .stButton > button {
            font-size: 18px;
            padding: 0.75em 2em;
            border-radius: 8px;
            background-color: #1f4e79;
            color: white;
            margin: 1em auto;
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("# Welcome to the Book Recommender 🎯")

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("📚 Recommend Books"):
        st.switch_page("pages/recommend_app.py")
with col2:
    if st.button("📖 Analyze Book"):
        st.switch_page("pages/books_app.py")
