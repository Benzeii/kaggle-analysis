import pandas as pd
import numpy
import sklearn
import sqlite3
from datetime import datetime
import time

# Debug: Confirm code version
print("Running code updated at 05:30 AM BST, July 12, 2025")

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

# Add budget squared as a feature
df['budget_squared'] = df['budget'] ** 2

# Print first 5 rows of cleaned columns with new feature
print(df[['title', 'budget', 'budget_squared', 'revenue', 'runtime', 'vote_average', 'year', 'main_genre']].head())

# Select final columns for modeling
final_columns = ['title', 'budget', 'budget_squared', 'revenue', 'runtime', 'vote_average', 'year', 'main_genre']
df_clean = df[final_columns]

# Check for outliers in budget and revenue
print("Budget stats:")
print(df_clean['budget'].describe())
print("\nBudget squared stats:")
print(df_clean['budget_squared'].describe())
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

# Import Scikit-learn for modeling and scaling
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Prepare features (X) and target (y), keep title for reference
X = df_clean[['budget', 'budget_squared', 'runtime', 'vote_average', 'year', 'main_genre']]
y = np.log1p(df_clean['revenue'])  # Log transform revenue
titles = df_clean['title']  # Store titles

# One-hot encode main_genre
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
genre_encoded = encoder.fit_transform(X[['main_genre']])
genre_columns = encoder.get_feature_names_out(['main_genre'])

# Prepare X_encoded with numerical features only for scaling
X_numeric = X[['budget', 'budget_squared', 'runtime', 'vote_average', 'year']].values
X_encoded_numeric = np.hstack((X_numeric, genre_encoded))

# Scale features with StandardScaler
scaler = StandardScaler()
X_train_numeric, X_test_numeric, y_train, y_test, titles_train, titles_test = train_test_split(X_numeric, y, titles, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled_numeric = scaler.transform(X_test_numeric)

# Combine scaled numeric features with one-hot encoded genres
X_train_scaled = np.hstack((X_train_scaled, encoder.transform(X.loc[titles_train.index][['main_genre']])))
X_test_scaled = np.hstack((X_test_scaled_numeric, encoder.transform(X.loc[titles_test.index][['main_genre']])))

# Train Gradient Boosting model with tuned parameters
model = GradientBoostingRegressor(n_estimators=200, max_depth=30, learning_rate=0.1, min_samples_leaf=2, min_samples_split=10, random_state=42)
print("Model parameters:", model.get_params())  # Debug: Verify parameters
start_time = time.time()
model.fit(X_train_scaled, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Predict on test set
y_pred_log = model.predict(X_test_scaled)

# Debug: Check raw predictions
print(f"Raw log predictions range: {np.min(y_pred_log):.4f} to {np.max(y_pred_log):.4f}")
print(f"Top 5 log predictions: {sorted(y_pred_log, reverse=True)[:5]}")  # Check highest values

# Reverse log transformation with adjusted clipping
y_pred_original = np.expm1(y_pred_log)
print(f"Max predicted value before clipping: {np.max(y_pred_original):.2e}")
y_pred_original = np.clip(y_pred_original, 1000, 5e10)  # Explicitly set to 50 billion
print(f"Max predicted value after clipping: {np.max(y_pred_original):.2e}")
y_test_original = np.expm1(y_test)

# Calculate mean squared error
mse_log = mean_squared_error(y_test, y_pred_log)
mse_original = mean_squared_error(y_test_original, y_pred_original)
print("Mean Squared Error (log scale):", mse_log)
print("Mean Squared Error (original scale):", mse_original)

# Save predictions to dataframe with titles
df_test = pd.DataFrame(X_test_scaled, columns=['budget', 'budget_squared', 'runtime', 'vote_average', 'year'] + list(genre_columns))
df_test['title'] = titles_test.reset_index(drop=True)
df_test['actual_revenue'] = y_test_original.reset_index(drop=True)
df_test['predicted_revenue'] = y_pred_original
df_test.to_csv('predictions.csv', index=False)

# Optional: Print some predictions for sanity check
print("Sample predictions:")
print(df_test[['title', 'actual_revenue', 'predicted_revenue']].head())

# SQLite setup with single connection
conn = sqlite3.connect('movies.db')
cursor = conn.cursor()

# Create movies table with updated columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        title TEXT,
        budget REAL,
        revenue REAL,
        runtime REAL,
        vote_average REAL,
        year INTEGER,
        main_genre TEXT,
        predicted_revenue REAL
    )
''')

# Load cleaned_movies.csv and insert into table
df_clean.to_sql('movies', conn, if_exists='replace', index=False)

# Check if predicted_revenue column exists, add if not
cursor.execute('PRAGMA table_info(movies)')
columns = [col[1] for col in cursor.fetchall()]
if 'predicted_revenue' not in columns:
    cursor.execute('ALTER TABLE movies ADD COLUMN predicted_revenue REAL')

# Load predictions.csv for update and verify
df_pred = pd.read_csv('predictions.csv')
print("Predictions CSV sample:")
print(df_pred[['title', 'predicted_revenue']].head())  # Verify saved predictions

# Merge predictions with test subset
df_test_subset = df_clean[df_clean['title'].isin(df_pred['title'])].copy()
df_merged = pd.merge(df_test_subset, df_pred[['title', 'predicted_revenue']], on='title', how='left')

# Update SQLite table with merged predictions
for idx, row in df_merged.iterrows():
    if pd.notna(row['predicted_revenue']):  # Only update if prediction exists
        cursor.execute('UPDATE movies SET predicted_revenue = ? WHERE title = ?', (row['predicted_revenue'], row['title']))

# Verify
cursor.execute('SELECT title, revenue, predicted_revenue FROM movies WHERE predicted_revenue IS NOT NULL LIMIT 5')
rows = cursor.fetchall()
print("Updated rows with predictions:")
for row in rows:
    print(row)

# Close connection
conn.commit()
conn.close()