'''import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
from io import BytesIO
import json

# Load Data
@st.cache_data
def load_data():
    word_freq_df = pd.read_parquet("/home/vaibhavi/spark-ml-venv/book_review_pipeline/data/book_word_frequencies.parquet")
    with open("/home/vaibhavi/spark-ml-venv/book_review_pipeline/data/top_100_books_data.json", "r") as f:
            books_metadata = json.load(f)["books"]

    return word_freq_df, books_metadata

word_freq_df, books_metadata = load_data()

st.set_page_config(layout="wide")
st.title("📚 Analyze Book")

col1, col2, col3 = st.columns([1, 2, 2])


# UI – Book selection

book_list = word_freq_df['title'].unique().tolist()

with col2:
    selected_book = st.selectbox("Choose a book to analyze", book_list)

# Get word frequencies
word_freq_row = word_freq_df[word_freq_df['title'] == selected_book]
word_freq = {}
if not word_freq_row.empty:
    word_freq = json.loads(word_freq_row['word_freq'].values[0])

# Get book metadata
book_info = next((book for book in books_metadata if book["title"] == selected_book), None)

print(type(books_metadata[0]))
print(books_metadata[0])


# Layout: 3 Columns

# ─── 📖 Column 1: Book Image ───────────────────────────
with col1:
    st.markdown("### 📖 Cover")
    if book_info and book_info.get("image"):
        try:
            response = requests.get(book_info["image"]["url"])
            img = BytesIO(response.content)
            st.image(img, use_container_width=True)
        except:
            st.warning("Image failed to load.")
    else:
        st.info("No image available.")

# ─── 📘 Column 2: Book Details ─────────────────────────
with col2:
    #selected_book = st.selectbox("Choose a book to analyze", book_list)
    st.markdown("### 📘 Book Details")
    if book_info:
        st.markdown(f"**Title:** {book_info['title']}")
        st.markdown(f"**Author:** {book_info['contributions'][0]['author']['name']}")
        st.markdown(f"**Release Year:** {book_info.get('release_year', 'N/A')}")
        st.markdown(f"**Rating:** ⭐ {round(book_info['rating'], 2)} from {book_info['ratings_count']} ratings")
        st.markdown(f"**Reviews Count:** 📝 {book_info['reviews_count']}")
        st.markdown("---")
        st.markdown("**Description:**")
        st.write(book_info['description'])
    else:
        st.warning("Metadata not found.")

# ─── ☁️ Column 3: Word Cloud & Rating Chart ───────────
with col3:
    st.markdown("### ☁️ Word Cloud")
    if word_freq:
        wc = WordCloud(width=500, height=300, background_color='white').generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No review data available.")

    # Rating Distribution
    if book_info and book_info.get("ratings_distribution"):
        st.markdown("### 📊 Ratings Breakdown")
        rating_data = book_info["ratings_distribution"]
        df_ratings = pd.DataFrame(rating_data)
        df_ratings = df_ratings.sort_values("rating")
        fig2, ax2 = plt.subplots()
        ax2.bar(df_ratings["rating"].astype(str), df_ratings["count"], width=0.6)
        ax2.set_xlabel("Rating")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
    else:
        st.info("No rating distribution data.")
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
from io import BytesIO
import json

# Load Data
@st.cache_data
def load_data():
    word_freq_df = pd.read_parquet("/home/vaibhavi/spark-ml-venv/book_review_pipeline/data/book_word_frequencies.parquet")
    with open("/home/vaibhavi/spark-ml-venv/book_review_pipeline/data/top_100_books_data.json", "r") as f:
        books_metadata = json.load(f)["books"]
    return word_freq_df, books_metadata

word_freq_df, books_metadata = load_data()

# Set page and inject CSS
st.set_page_config(layout="wide")

# Inject CSS for styling
st.markdown("""
    <style>
        /* Center the selectbox */
        div[data-baseweb="select"] {
            margin-left: auto;
            margin-right: auto;
            width: 60% !important;
        }

        /* Make select label center aligned */
        label[data-testid="stLabel"] > div {
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            color: #333333;
        }

        /* Customize fonts globally */
        html, body, [class*="css"]  {
            font-family: 'Playfair Display', sans-serif;
            font-size: 16px;
        }

        /* Title font */
        .css-10trblm.e16nr0p30 {
            font-family: 'Georgia', serif;
            font-size: 2.2rem;
            font-weight: bold;
            text-align: center;
        }

        /* Section headers */
        h3 {
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            color: #2C3E50;
        }
    </style>
""", unsafe_allow_html=True)


# Title
st.title("📚 Analyze Book")

# Centered Dropdown
book_list = word_freq_df['title'].unique().tolist()
st.markdown('<div class="centered-selectbox">', unsafe_allow_html=True)
selected_book = st.selectbox("Choose a book to analyze", book_list, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# Get word frequencies
word_freq_row = word_freq_df[word_freq_df['title'] == selected_book]
word_freq = {}
if not word_freq_row.empty:
    word_freq = json.loads(word_freq_row['word_freq'].values[0])

# Get book metadata
book_info = next((book for book in books_metadata if book["title"] == selected_book), None)

# Layout: 3 Columns in one row
col1, col2, col3 = st.columns([1.5, 2.5, 2])

# ─── 📖 Column 1: Book Image ───────────────────────────
with col1:
    st.subheader("📖 Cover")
    if book_info and book_info.get("image"):
        try:
            response = requests.get(book_info["image"]["url"])
            img = BytesIO(response.content)
            st.image(img, use_container_width=True)
        except:
            st.warning("Image failed to load.")
    else:
        st.info("No image available.")

# ─── 📘 Column 2: Book Details ─────────────────────────
with col2:
    st.subheader("📘 Book Details")
    if book_info:
        st.markdown(f"**Title:** {book_info['title']}")
        st.markdown(f"**Author:** {book_info['contributions'][0]['author']['name']}")
        st.markdown(f"**Release Year:** {book_info.get('release_year', 'N/A')}")
        st.markdown(f"**Rating:** ⭐ {round(book_info['rating'], 2)} from {book_info['ratings_count']} ratings")
        st.markdown(f"**Reviews Count:** 📝 {book_info['reviews_count']}")
        st.markdown("---")
        st.markdown("**Description:**")
        st.write(book_info['description'])
    else:
        st.warning("Metadata not found.")

# ─── ☁️ Column 3: Word Cloud & Rating Chart ───────────
with col3:
    st.subheader("☁️ Word Cloud")
    if word_freq:
        wc = WordCloud(width=500, height=300, background_color='white').generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No review data available.")

    # Rating Distribution
    if book_info and book_info.get("ratings_distribution"):
        st.subheader("📊 Ratings Breakdown")
        rating_data = book_info["ratings_distribution"]
        df_ratings = pd.DataFrame(rating_data)
        df_ratings = df_ratings.sort_values("rating")
        fig2, ax2 = plt.subplots()
        ax2.bar(df_ratings["rating"].astype(str), df_ratings["count"], width=0.6)
        ax2.set_xlabel("Rating")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
    else:
        st.info("No rating distribution data.")
