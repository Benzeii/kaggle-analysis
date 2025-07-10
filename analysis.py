import pandas as pd
import numpy
import sklearn
import sqlite3

df = pd.read_csv('tmdb_5000_movies.csv')


# Load the CSV file
df = pd.read_csv('tmdb_5000_movies.csv')

# Print first 5 rows
print(df.head())

# Print missing values per column
print(df.isnull().sum())

# Print data types of columns
print(df.dtypes)


# Remove rows with missing release_date
df = df.dropna(subset=['release_date'])

# Fill missing runtime with mean
df['runtime'] = df['runtime'].fillna(df['runtime'].mean())

# Remove rows where budget or revenue is 0
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Print new shape and missing values
print("Shape after cleaning:", df.shape)
print(df.isnull().sum())




