from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit

# 1. Merge ALS and content-based recommendations
def merge_hybrid_recommendations(
    als_recs: DataFrame,
    content_recs: DataFrame,
    weight_als: float = 0.3,
    weight_content: float = 0.7
) -> DataFrame:
    
    als_scored = als_recs.select("Title", col("rating").alias("als_score"))
    content_scored = content_recs.select("Title", col("similarity").alias("content_score"))

    

    # Inner join to ensure both have same book_id
    joined = als_scored.join(content_scored, on="Title", how="inner")

    # Weighted hybrid score
    final = joined.withColumn(
        "hybrid_score",
        (weight_als * col("als_score")) + (weight_content * col("content_score"))
    ).orderBy(col("hybrid_score").desc())

    return final
