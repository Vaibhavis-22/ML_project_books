import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexerModel

from rapidfuzz import process

# Initialize Spark
@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("BookRecs").getOrCreate()

spark = get_spark()

# Load necessary data
@st.cache_resource
def load_data():
    books = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/data/books_metadata/*", header=True)  # Or wherever your img & meta is
    features = spark.read.parquet("data/vectorized_books.parquet")
    als_model = ALSModel.load("models/als_model")

    return books, features, als_model

books_df, final_features_df, als_model = load_data()

from pyspark.sql.functions import col

def get_content_vector(title, features_df):
    result = features_df.filter(col("title") == title).select("final_features").collect()
    return result[0]['final_features'] if result else None


from als_model import get_als_recommendations
from content_model import get_content_recommendations
from hybrid_model import merge_hybrid_recommendations

# Sample: list of titles from your metadata DataFrame
all_titles = books_df.select("title").rdd.flatMap(lambda x: x).collect()

def get_best_match(user_input, titles, cutoff=70):
    match = process.extractOne(user_input, titles, score_cutoff=cutoff)
    return match[0] if match else None


# Streamlit UI
st.title("📚 Hybrid Book Recommender")

given_book = st.text_input("Enter the book name ")
# Streamlit input


matched_title = get_best_match(given_book, all_titles)
if matched_title:
    st.success(f"Using closest match: **{matched_title}**")
    # Call your content-based recommendation function with matched_title
else:
    st.warning("No close match found. Please check the title spelling.")
user_id = st.text_input("Enter your user id ")

if st.checkbox("Show available user IDs"):
    user_indexer = StringIndexerModel.load("models/user_indexer_fitted")
    st.write(user_indexer.labels)


if st.button("Recommend Books"):
    st.subheader("🔍 Generating Recommendations...")

    # Get user's top rated book as proxy for content-based input
    als_recs = get_als_recommendations(user_id, spark)

    if als_recs is not None and not als_recs.rdd.isEmpty():
        #top_rated = als_recs.limit(5).collect()[0]['original_book_id']

        # Fetch metadata
        als_recs_with_titles = als_recs.join(books_df, als_recs.original_book_id == books_df.Id, "left")

        display_df = als_recs_with_titles.select("Title").distinct()
        
        titles = display_df.select("Title").rdd.flatMap(lambda x: x).collect()
        st.write("ALS Recommendations:")
        

        for title in titles:
            st.write(f"- {title}")


        #st.write(f"Top recommended book ID: {top_rated}")
    else:
        st.warning("No recommendations found for this user.")

    
    try :

        input_vec = final_features_df.filter(final_features_df.Title == matched_title).select("final_features").collect()[0]['final_features']
    

        content_recs = get_content_recommendations(final_features_df,input_vec)
    

        # Merge
        target_vector = get_content_vector(titles, final_features_df)

        hybrid_recs = merge_hybrid_recommendations(als_recs_with_titles, content_recs)

    # Join with book info
    
        hybrid_titles = hybrid_recs.select("Title").rdd.flatMap(lambda x: x).collect()
        st.subheader("📌 Hybrid Recommendations")


        for title in hybrid_titles:
            st.write(f"- {title}")

    except:
        st.write("--------------------------------")
 


 
