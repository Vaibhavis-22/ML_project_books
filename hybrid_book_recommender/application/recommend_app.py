import streamlit as st
from pyspark.sql import SparkSession
from rapidfuzz import process
from pyspark.sql.functions import col

from als_recommendations import get_als_recommendations
from content_recommendations import get_content_recommendations
#from hybrid_model import merge_hybrid_recommendations

# Initialize Spark
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("BookRecs").getOrCreate()
        #.config("spark.driver.memory", "2g") \
        #.config("spark.executor.memory", "2g") \

spark = get_spark()

# Load necessary data
@st.cache_resource
def load_data():
    books = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/hybrid_book_recommender/data/books_df")
    #pca_vector_features
    features = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/hybrid_book_recommender/data/pca_vectorized_df")
    users_df = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/hybrid_book_recommender/data/merged_interactions.parquet") \
                       .select("user_id").distinct()
    return books, features, users_df

books_df, final_features_df,users_df = load_data()

# Prepare list of all book titles
all_titles = books_df.select("title").rdd.flatMap(lambda x: x).collect()

# Fuzzy matcher
def get_best_match(user_input, titles, cutoff=70):
    match = process.extractOne(user_input, titles, score_cutoff=cutoff)
    return match[0] if match else None


# UI
st.title("📚 Hybrid Book Recommender")

given_book = st.text_input("Enter the book name:")
matched_title = get_best_match(given_book, all_titles)

if matched_title:
    st.success(f"Using closest match: **{matched_title}**")
else:
    st.warning("No close match found. Please check the title spelling.")

user_id = st.text_input("Enter user_id")

if st.checkbox("Show available user IDs"):
    st.write(users_df.select("user_id").distinct().limit(50).toPandas())


if st.button("Recommend Books") and matched_title:
    st.subheader("🔍 Generating Recommendations...")

    # Content-based Recommendations
    content_recs = get_content_recommendations(matched_title, final_features_df, spark)

    if content_recs is not None and not content_recs.rdd.isEmpty():
        #content_recs_with_titles = content_recs.join(books_df, content_recs.title == books_df.title, "left")
        content_titles = content_recs.select("title").rdd.flatMap(lambda x: x).collect()

        st.subheader("🟢 Content-Based Recommendations:")
        for title in content_titles:
            st.write(f"- {title}")
    else:
        st.warning("No content-based recommendations found.")

    # ALS Recommendations
    als_recs = get_als_recommendations(user_id, spark)

    if als_recs is not None and not als_recs.rdd.isEmpty():
        als_recs_with_titles = als_recs.join(books_df, als_recs.book_id == books_df.id, "left")

        display_df = als_recs_with_titles.select("title").distinct()
        als_titles = display_df.rdd.flatMap(lambda x: x).collect()

        st.subheader("🔵 ALS Recommendations:")
        for title in als_titles:
            st.write(f"- {title}")
    else:
        st.warning("No ALS recommendations found for this user.")


