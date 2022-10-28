#!pip install pandas==1.3.5
import pickle
import pandas as pd
import sys
import os
os.environ['PYSPARK_DRIVER_PYTHON_OPTS']= "notebook"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
import string
import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType
import warnings

from recommenders.utils.spark_utils import start_or_get_spark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder


warnings.simplefilter(action='ignore', category=FutureWarning)

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))

spark = SparkSession \
    .builder \
    .appName("Example") \
    .config("spark.driver.memory", "16g") \
    .getOrCreate()



print(pd.__version__)

ratings_data = pd.read_csv(
        r"C:\Users\mirza\Desktop\dataset\MovieLens-1M\ratings.csv",sep = "::",
        names=["userId", "movieId", "ratings", "timestamp"])

movie= pd.read_csv(
        r"C:\Users\mirza\Desktop\dataset\MovieLens-1M\movies.csv",sep = "::",
        names=["movieId", "Title", "Genres"])

users = ratings_data["userId"].unique()

ratings_data=spark.createDataFrame(ratings_data)



(training, test) = ratings_data.randomSplit ([0.8, 0.2])

als = ALS(userCol="userId", itemCol="movieId", ratingCol="ratings", coldStartStrategy="drop", nonnegative=True)

param_grid = ParamGridBuilder()\
            .addGrid(als.rank, [10])\
            .addGrid(als.maxIter, [15])\
            .addGrid(als.regParam, [0.05])\
            .build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings",
                               predictionCol="prediction")

tvs = TrainValidationSplit(estimator = als,
                          estimatorParamMaps= param_grid,
                          evaluator = evaluator)

model = tvs.fit(training)

best_model = model.bestModel

predictions = best_model.transform(test)
rmse = evaluator.evaluate(predictions)

print("RMSE = " + str(rmse))
print("**Best Model**")
print(" Rank:"), best_model.rank
print(" MaxIter:"), best_model._java_obj.parent().getMaxIter()
print(" RegParam:"), best_model._java_obj.parent().getRegParam()


user_recs = best_model.recommendForAllUsers(5)

user_recs.show()


movie["Genres"]= movie["Genres"].str.replace('[{}]'.format(string.punctuation), ", ")
movie["movieId"]= movie["movieId"].str.replace('[{}]'.format(string.punctuation), '')
movie["movieId"]=movie["movieId"].astype(int)



pickle.dump(movie, open("movie.pkl","wb" ))
pickle.dump(users, open("users.pkl","wb" ))
user_recs.select("userId", "recommendations").write.save("user_recs.parquet")

#pickle.dump(movie, open("movie.pkl","wb" ))
#pickle.dump(users, open("users.pkl","wb" ))
#user_recs.select("userId", "recommendations").write.save("user_recs.parquet")
