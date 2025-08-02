from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSH, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors, VectorUDT

def train_and_save_lsh_model(spark, vectorized_df, lsh_model_path, pca_features_path):
    """
    Trains and saves an LSH model on dimension-reduced vectors for scalable similarity search.

    Args:
        spark (SparkSession): The active Spark session.
        vectorized_df (DataFrame): The DataFrame with the pre-computed 'features_w2v' column.
        lsh_model_path (str): The file path to save the fitted LSH model and PCA model.
        pca_features_path (str): The file path to save the DataFrame with reduced PCA features.
    """
    try:
        print("Starting LSH model training with PCA for dimensionality reduction.")

        # --- 1. Define and Fit the PCA Model ---
        # PCA takes your large 'features_w2v' vector and reduces its size.
        print("Step 1: Defining and fitting a PCA model to reduce vector dimensions.")
        pca = PCA(k=150, inputCol="features_w2v", outputCol="pca_features")
        pca_model = pca.fit(vectorized_df)
        print("PCA model fitted successfully. Original vector dimensions reduced to 150.")
        
        # Transform the DataFrame to get the new, smaller feature vectors. This is the only transform call.
        pca_features_df = pca_model.transform(vectorized_df)
        
        # --- 2. Save the new DataFrame with PCA Features ---
        print(f"Step 2: Saving the DataFrame with reduced PCA features to {pca_features_path}")
        pca_features_df.select("id", "title", "pca_features").write.mode("overwrite").parquet(pca_features_path)
        print("PCA features DataFrame saved.")

        # --- 3. Define and Fit the LSH Model on PCA Vectors ---
        print("Step 3: Defining and fitting the LSH model on the new PCA vectors.")
        brp_lsh = BucketedRandomProjectionLSH(
            inputCol="pca_features",  # Use the new, smaller PCA vector
            outputCol="hashes",
            numHashTables=10,
            bucketLength=2.0
        )
        
        # The LSH model is now trained on the smaller, more efficient vectors.
        lsh_model = brp_lsh.fit(pca_features_df)
        print("LSH model fitted successfully.")

        # --- 4. Save the Fitted PCA and LSH Models ---
        # We need to save both the PCA model (to transform new vectors) and the LSH model.
        print(f"Step 4: Saving the fitted PCA and LSH models to {lsh_model_path}")
        
        # We'll save them as part of a single pipeline for modularity
        final_model_pipeline = Pipeline(stages=[pca_model, lsh_model]).fit(vectorized_df)
        final_model_pipeline.write().overwrite().save(lsh_model_path)
        print("PCA and LSH models saved as a single pipeline model.")
        
    except Exception as e:
        print(f"An error occurred during LSH model training/saving: {e}")

def run_example_training():
    """
    Simulates the full training and saving process.
    """
    # Use a safer memory configuration to prevent OOM during the read operation
    spark = SparkSession.builder \
        .appName("LSHModelTrainerWithPCA") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()
    
    # --- Load your existing vectorized DataFrame from its location ---
    vectorized_df = spark.read.parquet('/home/vaibhavi/spark-ml-venv/ml_project/data/data_modelling/output/books_vectorized')

    # Define the path to save the models and the new feature data.
    lsh_model_path = "model/lsh_pca_model"
    pca_features_path = "data/pca_vectorized_df"

    # Run the training and saving function
    train_and_save_lsh_model(spark, vectorized_df, lsh_model_path, pca_features_path)

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    run_example_training()
