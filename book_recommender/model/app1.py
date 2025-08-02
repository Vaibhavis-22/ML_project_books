import streamlit as st
from pyspark.sql import SparkSession
from rapidfuzz import process
from pyspark.sql.functions import col

from als_model import get_als_recommendations
from content_recs import get_content_recommendations
from hybrid_model import merge_hybrid_recommendations

# Initialize Spark
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("BookRecs") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

spark = get_spark()

# Load necessary data
@st.cache_resource
def load_data():
    books = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/data/clean_books.parquet")
    features = spark.read.parquet("data/vectorized_books.parquet")
    users_df = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/model/data/merged_interactions.parquet") \
                        .select("user_id").distinct()
    return books, features, users_df

books_df, final_features_df, users_df = load_data()

# Prepare list of all book titles
all_titles = books_df.select("title").rdd.flatMap(lambda x: x).collect()

# Fuzzy matcher
def get_best_match(user_input, titles, cutoff=70):
    match = process.extractOne(user_input, titles, score_cutoff=cutoff)
    return match[0] if match else None


from pyspark.sql.functions import col
from pyspark.sql.types import LongType, DoubleType, StringType

def preprocess_books_df(df):
    # Rename 'id' to 'book_id'
    df = df.withColumnRenamed("id", "book_id")

    # Cast book_id to string (if IDs are numeric strings, you can cast to LongType)
    df = df.withColumn("book_id", col("book_id").cast(StringType()))

    # Cast ratings and counts to proper types
    df = df.withColumn("rating_value", col("rating_value").cast(DoubleType()))
    df = df.withColumn("rating_count", col("rating_count").cast(LongType()))

    # Drop rows with null book_id (cannot join on null)
    df = df.filter(col("book_id").isNotNull())

    # Select only needed columns to optimize
    df = df.select(
        "book_id",
        "title",
        "author_name",
        "rating_value",
        "rating_count",
        "genre_tag",
        "image_url"
    )
    return df

books_df = preprocess_books_df(books_df)

# UI
st.title("📚 Hybrid Book Recommender")

given_book = st.text_input("Enter the book name:")
matched_title = get_best_match(given_book, all_titles)

if matched_title:
    st.success(f"Using closest match: **{matched_title}**")
else:
    st.warning("No close match found. Please check the title spelling.")

user_id_input = st.text_input("Enter your user ID:")
user_id = int(user_id_input) if user_id_input.isdigit() else None

if st.checkbox("Show available user IDs"):
    st.write(users_df.select("user_id").distinct().limit(50).toPandas())

if st.button("Recommend Books") and matched_title and user_id is not None:
    st.subheader("🔍 Generating Recommendations...")

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

    # Content-based Recommendations
    content_recs = get_content_recommendations(matched_title, final_features_df, spark)

    if content_recs is not None and not content_recs.rdd.isEmpty():
        content_recs_with_titles = content_recs.join(books_df, content_recs.Title == books_df.title, "left")
        content_titles = content_recs_with_titles.select("title").rdd.flatMap(lambda x: x).collect()

        st.subheader("🟢 Content-Based Recommendations:")
        for title in content_titles:
            st.write(f"- {title}")
    else:
        st.warning("No content-based recommendations found.")

    # Hybrid Recommendations (if both exist)
    if als_recs is not None and content_recs is not None:
        try:
            hybrid_recs = merge_hybrid_recommendations(als_recs_with_titles, content_recs)
            hybrid_titles = hybrid_recs.select("title").rdd.flatMap(lambda x: x).collect()

            st.subheader("📌 Hybrid Recommendations:")
            for title in hybrid_titles:
                st.write(f"- {title}")
        except Exception as e:
            st.error(f"Error merging hybrid recommendations: {e}")

else:
    st.info("Please enter both a valid book name and numeric user ID.")
