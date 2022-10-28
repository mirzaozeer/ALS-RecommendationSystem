import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
import pickle
import pyspark
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession \
  .builder \
  .appName("Example") \
  .config("spark.driver.memory", "16g")\
  .getOrCreate()



movie = pickle.load(open("movie.pkl", "rb"))
users = pickle.load(open("users.pkl", "rb"))
user_recs = spark.read.load(r"C:\Users\mirza\user_recs.parquet")


def fetch_poster(movie_id):

  try:
    response = requests.post("https://api.themoviedb.org/3/movie/{}?api_key=ce3e3e03053ce84779d05fc694470a6d&language=en-US".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

  except:
    return "static\logo.jpg"




def recommend(userıd):
  recs = user_recs.filter(user_recs.userId==int(userıd))
  recs = recs.select("recommendations.movieId","recommendations.rating")  # add "userId" ıf you want to add userId column
  movies = recs.select("movieId").toPandas().iloc[0, 0]
  ratings_matrix = pd.DataFrame(movies, columns=["movieId"])
  matrix = pd.merge(ratings_matrix, movie, how="inner", on="movieId")

  recommended_movies = []
  recommended_movies_posters = []

  for k in matrix.Title:
    recommended_movies.append(k)

  for i in str(matrix.movieId):
    recommended_movies_posters.append(fetch_poster(i))

  return recommended_movies , recommended_movies_posters


app = Flask(__name__)

@app.route("/")
def man():
  return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def home():
  user = request.form["userıd"]
  names, posters = recommend(user)
  return render_template("after.html", names=names, posters=posters, user=user)



if __name__ == "__main__":
    app.run(port=3000, debug=True)