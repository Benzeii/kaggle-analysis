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



# Convert release_date to datetime and extract year
df['year'] = pd.to_datetime(df['release_date']).dt.year

# Drop release_date column
df = df.drop(columns=['release_date'])

# Extract first genre from genres (JSON-like string)
import json
def get_first_genre(genre_str):
    genres = json.loads(genre_str)
    return genres[0]['name'] if genres else 'Unknown'

df['main_genre'] = df['genres'].apply(get_first_genre)

# Drop genres column
df = df.drop(columns=['genres'])

# Print first 5 rows of cleaned columns
print(df[['title', 'budget', 'revenue', 'runtime', 'vote_average', 'year', 'main_genre']].head())

# Select final columns for modeling
final_columns = ['title', 'budget', 'revenue', 'runtime', 'vote_average', 'year', 'main_genre']
df_clean = df[final_columns]

# Check for outliers in budget and revenue
print("Budget stats:")
print(df_clean['budget'].describe())
print("\nRevenue stats:")
print(df_clean['revenue'].describe())

# Save cleaned dataframe to CSV
df_clean.to_csv('cleaned_movies.csv', index=False)

# Print final shape
print("Final cleaned shape:", df_clean.shape)

# Filter out rows with budget or revenue < $1000
df_clean = df_clean[(df_clean['budget'] >= 1000) & (df_clean['revenue'] >= 1000)]

# Print new shape after outlier removal
print("Shape after outlier removal:", df_clean.shape)

# Import Scikit-learn for modeling
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare features (X) and target (y)
X = df_clean[['budget', 'runtime', 'vote_average', 'year']]
y = df_clean['revenue']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save predictions to dataframe
df_test = X_test.copy()
df_test['actual_revenue'] = y_test
df_test['predicted_revenue'] = y_pred
df_test.to_csv('predictions.csv', index=False)
