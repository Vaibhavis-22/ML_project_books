from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import lit



'''def get_als_recommendations(user_id, spark):
    try:
        print(f"Getting ALS recs for user: {user_id}")
        
        user_indexer = StringIndexerModel.load("models/user_indexer_fitted")
        als_model = ALSModel.load("models/als_model")
        
        user_df = spark.createDataFrame([(user_id,)], ["User_id"])
        user_df.show()

        user_transformed = user_indexer.transform(user_df)
        user_transformed.show()

        if user_transformed.select("user").rdd.isEmpty():
            print("User not found in StringIndexer model.")
            return None

        recs = als_model.recommendForUserSubset(user_transformed.select("user"), 10)
        recs.show()

        if recs.rdd.isEmpty():
            print("ALS returned no recommendations.")
            return None

        recs = recs.selectExpr("explode(recommendations) as rec").selectExpr("rec.book_id", "rec.rating")
        recs.show()
        
        return recs

    except Exception as e:
        print(f"Error in ALS recommendation: {e}")
        return None'''

from pyspark.ml.feature import IndexToString

book_indexer = StringIndexerModel.load("models/book_indexer_fitted")

def get_als_recommendations(user_id, spark):
    try:
        user_indexer = StringIndexerModel.load("models/user_indexer_fitted")
        labels = user_indexer.labels  # list of seen user_ids as strings

        if str(user_id) not in labels:
            print("User ID not found in training data.")
            return None

        user_df = spark.createDataFrame([(user_id,)], ["User_id"])
        user_transformed = user_indexer.transform(user_df)

        recs = ALSModel.load("models/als_model").recommendForUserSubset(user_transformed.select("user"), 10)

        if recs.rdd.isEmpty():
            return None

        recs = recs.selectExpr("explode(recommendations) as rec").selectExpr("rec.book_id", "rec.rating")

        converter = IndexToString(inputCol="book_id", outputCol="original_book_id")
        converter.setLabels(book_indexer.labels)

        recs = converter.transform(recs)

        return recs

    except Exception as e:
        print(f"Error in ALS recommendation: {e}")
        return None
