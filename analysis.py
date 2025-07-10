import pandas as pd
import numpy
import sklearn
import sqlite3

df = pd.read_csv('tmdb_5000_movies.csv')

# Print first 5 rows
print(df.head())

# Print missing values per column
print(df.isnull().sum())

# Print data types of columns
print(df.dtypes)