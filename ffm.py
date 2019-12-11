import random
import pandas as pd
import numpy as np
# from hashlib import sha256

from pyspark.sql import types
from pyspark.sql import SQLContext
from pyspark.sql.functions import isnan
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import pyspark

import time
import numpy as np
# import matplotlib.pyplot as plt

# set the seed
np.random.seed(1)

app_name = "final_project_notebook"
# master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext

#sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)
# sc = spark.sparkContext

train_parquet = spark.read.parquet("gs://w261-f19-project-team15/data/train.parquet")

cate_field_start = 14
cate_field_end = 40

#rename files and recast integer types on the numeric features

oldColNames = train_parquet.schema.names

train_parquet = train_parquet.withColumn("label", train_parquet["_c0"])
for colNum in range(1,cate_field_start): 
    colName = "_c" + str(colNum)
    train_parquet = train_parquet.withColumn("int_feature_"+ str(colNum), train_parquet[colName].cast(types.IntegerType()))
for colNum in range(cate_field_start,cate_field_end): 
    colName = "_c" + str(colNum)
    train_parquet = train_parquet.withColumn("cate_feature_"+ str(colNum-cate_field_start+1), train_parquet[colName])

#drop the old columns
adjusted_labels_train_parquet = train_parquet.drop(*oldColNames)

intFieldNames = [colName for colName, dType in adjusted_labels_train_parquet.dtypes if dType == 'int']
cateFieldNames = [colName for colName, dType in adjusted_labels_train_parquet.dtypes if dType == 'string' and colName != 'label']


# Create thresholds
threshold = 10

train_parquet_MD = adjusted_labels_train_parquet

for col in cateFieldNames:
    valuesToKeep = adjusted_labels_train_parquet.groupBy(col).count().filter(f"count >= {threshold}").select(col)
    valuesToKeep = valuesToKeep.withColumn("_"+col, adjusted_labels_train_parquet[col])
    valuesToKeep = valuesToKeep.drop(col)

    train_parquet_MD = train_parquet_MD.join(F.broadcast(valuesToKeep), train_parquet_MD[col] == valuesToKeep["_"+col], 'leftouter')
    train_parquet_MD = train_parquet_MD.withColumn(col, F.when(F.col("_"+col).isNull(), "***").otherwise(F.col("_"+col)))
    train_parquet_MD = train_parquet_MD.drop("_"+col)
    
train_parquet_reduced_dimensions = train_parquet_MD

# Bin numeric variables
for col in intFieldNames:
    train_parquet_reduced_dimensions = train_parquet_reduced_dimensions.withColumn(col, F.floor(F.log(F.col(col) + F.lit(2))))

# Hash Features
n_features = 50000
n_fields = len(intFieldNames) + len(cateFieldNames)

from pyspark.ml.feature import FeatureHasher
hasher = FeatureHasher()
hasher.setCategoricalCols(intFieldNames)
hasher.setNumFeatures(n_features)

for col in intFieldNames + cateFieldNames:
    hasher.setInputCols([col])
    hasher.setOutputCol(col+"_hashed")
    train_parquet_reduced_dimensions = hasher.transform(train_parquet_reduced_dimensions)
    
hashed_columns = train_parquet_reduced_dimensions.schema.names[-n_fields:]

# Convert values to hashes
def parse_sparse_vectors(vector, field_ind):
    if vector.indices.size > 0:
        return int(vector.indices[0])
    else:
        return None

vector_parser = F.udf(parse_sparse_vectors, types.IntegerType())

train_parquet_hashed = train_parquet_reduced_dimensions
for field_ind, col in enumerate(hashed_columns):
    
    train_parquet_hashed = train_parquet_hashed.withColumn(col, vector_parser(col, F.lit(field_ind)))

train_parquet_hashed = train_parquet_hashed.drop(*(intFieldNames + cateFieldNames))

# Change labels to be -1 and 1
train_parquet_hashed = train_parquet_hashed.withColumn("label", F.when(F.col("label") == 0, -1).otherwise(F.col("label")))

def phi(x, W):
    total = 0
    for i in range(len(x) - 1):
        if not x[i]:
            continue
            
        for j in range(i + 1, len(x)):
            if x[j]:
                total += np.dot(W[int(x[i]), j, :], W[int(x[j]), i, :])
                            
    return total

def kappa(y, features, W):
    return -int(y)/(1 + np.exp(y*phi(features, W)))

# Test Train Split
train_df, test_df = train_parquet_hashed.randomSplit([0.8, 0.2])
batches = train_df.randomSplit([0.01] * 100)

# Initialize model parameters
k = 2
n_features = 50000
n_fields = 39
eta = 0.3
reg_c = 0.1
sc.broadcast(k)
sc.broadcast(n_features)
sc.broadcast(n_fields)
sc.broadcast(reg_c)
sc.broadcast(eta)

def gradient(x, W):
    y = int(x[0])
    features = x[1:]
    kap = kappa(y, features, W)
    
    for i in range(len(features) - 1):
        if not features[i]:
            continue
            
        for j in range(i+1, len(features)):
            if features[j]:
                yield ((features[i], j), (kap * W[int(features[j]), i, :], 1))
                yield ((features[j], i), (kap * W[int(features[i]), j, :], 1))
                
def log_loss(dataRDD, W):
    return dataRDD.map(lambda x: np.log(1 + np.exp(-int(x[0]) * phi(x[1:], W)))).mean()

def gd_update(dataRDD, W):
    grad = dataRDD.flatMap(lambda x: gradient(x, W))\
                .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
                .map(lambda x: ((x[0][0], x[0][1]), x[1][0] / x[1][1]))\
                .collect()
    
    grad_update = np.zeros(shape=(n_features, n_fields, k))
    
    for indices, vector in grad:
        feature_index = indices[0]
        field_index = indices[1]
        
        grad_update[int(feature_index), field_index, :] += vector
    
    new_model = W - eta * grad_update
    
    return new_model

def gradient_descent(split_data, w_init, n_steps = 10):
    
    model = sc.broadcast(w_init)

    start = time.time()
    for i in range(n_steps):
        train_rdd = split_data[i].rdd
        print("----------")
        print(f"STEP: {i+1}")
        new_model = gd_update(train_rdd, model.value)
        model = sc.broadcast(new_model)
        train_loss = log_loss(train_rdd, model.value)
        print(f"Training Loss: {train_loss}")
        test_loss = log_loss(split_data[i + 50].rdd, model.value)
        print(f"Test Loss: {test_loss}")
    print(f"\n... trained {n_steps} iterations in {time.time() - start} seconds")

np.random.seed(1)
w_init = np.random.uniform(0, 1/np.sqrt(k), size=(n_features, n_fields, k))
gradient_descent(batches, w_init)