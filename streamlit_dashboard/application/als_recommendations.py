from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode

def get_als_recommendations(user_id, spark):
    try:
        als_model = ALSModel.load("/home/vaibhavi/spark-ml-venv/ml_project/hybrid_book_recommender/models/als_model")

        # Column name must match training ("user")
        user_df = spark.createDataFrame([(user_id,)], ["user"])

        recs = als_model.recommendForUserSubset(user_df, 25)

        if recs.rdd.isEmpty():
            return None

        recs = recs.select(explode("recommendations").alias("rec"))
        recs = recs.selectExpr("rec.item as book_id", "rec.rating")

        return recs

    except Exception as e:
        print(f"Error in ALS recommendation: {e}")
