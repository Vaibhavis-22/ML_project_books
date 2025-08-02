from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .appName("BookRecs") \
        .getOrCreate()

books = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/data/clean_books.parquet")

books.printSchema()

features = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/data/vectorized_books.parquet")

features.printSchema()

users_df = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/model/data/merged_interactions.parquet").select("user_id").distinct()

users_df.printSchema()