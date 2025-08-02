import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Trial").getOrCreate()

books = spark.read.parquet("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/model/data/merged_interactions.parquet")  # Or wherever your img & meta is

books.printSchema()