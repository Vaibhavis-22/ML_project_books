import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Trial").getOrCreate()

books = spark.read.csv("/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/data/books_metadata/*")  # Or wherever your img & meta is

books.printSchema()