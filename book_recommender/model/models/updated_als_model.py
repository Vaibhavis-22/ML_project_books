from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os

def train_and_save_als_model(spark, interaction_data_path, als_model_path):
    """
    Trains and saves an ALS recommendation model on user interaction data.

    Args:
        spark (SparkSession): The active Spark session.
        interaction_data_path (str): The file path to the user interaction data (e.g., Parquet file).
        als_model_path (str): The directory where the trained ALS model will be saved.
    """
    try:
        print("Starting ALS model training process.")
        
        # --- 1. Load the User Interaction Data ---
        # Load the merged Parquet file. This is your user-item interaction data.
        print(f"Loading user interaction data from {interaction_data_path}...")
        df = spark.read.parquet(interaction_data_path)
        print("Data loaded successfully.")

        # --- 2. Clean and Prepare Data for ALS ---
        # The ALS model requires user, item, and rating columns to be of numeric type.
        # Ensure column names match your data's schema.
        # This is a critical step to ensure the model runs correctly.
        df_als = df.selectExpr("user_id AS user", "book_id AS item", "rating AS rating")
        
        # Drop any rows with nulls, as ALS cannot handle them.
        df_als = df_als.dropna()
        
        print("Data prepared for ALS model training.")
        df_als.printSchema()
        df_als.show(5)

        # --- 3. Build and Configure the ALS Model ---
        # Set hyperparameters for the ALS model. These can be tuned for better performance.
        als = ALS(
            userCol="user",
            itemCol="item",
            ratingCol="rating",
            # ALS is sensitive to implicit feedback, so we set this to false for explicit ratings.
            implicitPrefs=False,
            # 'rank' is the number of latent factors, a key hyperparameter.
            rank=10,
            # 'regParam' is the regularization parameter to prevent overfitting.
            regParam=0.1,
            # 'nonnegative=True' ensures that user/item latent factors are non-negative.
            nonnegative=True,
            # 'coldStartStrategy="drop"' drops any rows with new users/items during prediction.
            coldStartStrategy="drop"
        )
        print("ALS model defined with hyperparameters.")
        
        # --- 4. Train the ALS Model ---
        # This is a heavy computation task.
        print("Training the ALS model. This may take some time...")
        als_model = als.fit(df_als)
        print("ALS model trained successfully.")

        # --- 5. Save the Trained Model ---
        # Saving the model allows us to load it later for making predictions.
        print(f"\nSaving the trained ALS model to {als_model_path}...")
        als_model.write().overwrite().save(als_model_path)
        print("ALS model saved.")

    except Exception as e:
        print(f"An error occurred during ALS model training: {e}")

if __name__ == "__main__":
    # Define your Spark session with a robust memory configuration
    spark = SparkSession.builder \
        .appName("ALSModelTrainer") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .master("local[*]") \
        .getOrCreate()
    
    # Define the paths for your data and where the model will be saved
    interaction_data_path = "/home/vaibhavi/spark-ml-venv/ml_project/book_recommender/model/data/merged_interactions.parquet"
    als_model_path = "model/als_model"

    # Run the training and saving function
    train_and_save_als_model(spark, interaction_data_path, als_model_path)

    # Stop the Spark session
    spark.stop()
