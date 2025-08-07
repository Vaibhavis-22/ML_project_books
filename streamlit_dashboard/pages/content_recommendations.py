from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql.functions import col

def get_content_recommendations(book_title, vector_df, spark, top_n=10):
    try:
        print(f"Getting content-based recommendations for: {book_title}")

        # Load LSH model
        lsh_model = BucketedRandomProjectionLSHModel.load("/home/vaibhavi/spark-ml-venv/ml_project/hybrid_book_recommender/models/lsh_pca_model")

        # Get vector for the given book title
        book_vector = vector_df.filter(col("title") == book_title).select("title", "pca_features")

        if book_vector.rdd.isEmpty():
            print("Book title not found in vector dataset.")
            return None

        # Get the vector as a key
        key_vector = book_vector.first()["pca_features"]

        # Get approximate nearest neighbors
        similar_books = lsh_model.approxNearestNeighbors(vector_df, key_vector, top_n + 1)

        # Exclude the input book from results
        recommendations = similar_books.filter(col("title") != book_title).select("title", "distCol")

        return recommendations

    except Exception as e:
        print(f"Error in content-based recommendation: {e}")
        return None
