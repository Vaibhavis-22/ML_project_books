from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSH

def train_and_save_lsh_from_pca(pca_features_path, lsh_model_path):
    spark = SparkSession.builder \
        .appName("TrainLSHModel") \
        .master("local[*]") \
        .getOrCreate()

    # Load PCA features
    df_pca = spark.read.parquet(pca_features_path) 

    # Define and train LSH
    lsh = BucketedRandomProjectionLSH(
        inputCol="pca_features",
        outputCol="hashes",
        numHashTables=10,
        bucketLength=2.0
    )
    lsh_model = lsh.fit(df_pca)

    # Save only the LSH model
    lsh_model.write().overwrite().save(lsh_model_path)
    print(f"LSH model saved at: {lsh_model_path}")

    spark.stop()

if __name__ == "__main__":
    pca_features_path = "/home/vaibhavi/spark-ml-venv/ml_project/hybrid_book_recommender/data/pca_vectorized_df"
    lsh_model_path = "/home/vaibhavi/spark-ml-venv/ml_project/hybrid_book_recommender/models/lsh_pca_model"
    train_and_save_lsh_from_pca(pca_features_path, lsh_model_path)
