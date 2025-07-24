import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# 1. Cosine similarity handling SparseVector
def compute_cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    vec1_np = vec1.toArray() if isinstance(vec1, (SparseVector, DenseVector)) else np.array(vec1)
    vec2_np = vec2.toArray() if isinstance(vec2, (SparseVector, DenseVector)) else np.array(vec2)
    
    dot_product = float(np.dot(vec1_np, vec2_np))
    norm_product = np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
    return float(dot_product / norm_product) if norm_product != 0 else 0.0

# 2. Return UDF that locks in input vector
def get_cosine_similarity_udf(input_vec):
    def _cosine(v):
        return compute_cosine_similarity(v, input_vec)
    return udf(_cosine, DoubleType())

# 3. Main recommendation logic
def get_content_recommendations(df_vectorized, input_vector, top_n=10):
    similarity_udf = get_cosine_similarity_udf(input_vector)
    df_with_sim = df_vectorized.withColumn("similarity", similarity_udf("final_features"))
    return df_with_sim.orderBy("similarity", ascending=False).limit(top_n)
