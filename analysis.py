import pandas as pd
import numpy
import sklearn
import sqlite3

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

# Import Scikit-learn for modeling and scaling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Prepare features (X) and target (y), keep title for reference
X = df_clean[['budget', 'runtime', 'vote_average', 'year', 'main_genre']]
y = np.log1p(df_clean['revenue'])  # Log transform revenue
titles = df_clean['title']  # Store titles

# One-hot encode main_genre
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
genre_encoded = encoder.fit_transform(X[['main_genre']])
genre_columns = encoder.get_feature_names_out(['main_genre'])
X_encoded = np.hstack((X[['budget', 'runtime', 'vote_average', 'year']].values, genre_encoded))

# Split data into training and test sets (80% train, 20% test) before scaling
X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split(X_encoded, y, titles, test_size=0.2, random_state=42)

# Scale features separately for train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler for test

# Train Random Forest model with increased trees
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_log = model.predict(X_test_scaled)

# Reverse log transformation with adjusted clipping
y_pred_original = np.expm1(y_pred_log)
y_pred_original = np.clip(y_pred_original, 1000, 1e10)  # Relaxed upper bound to 10 billion
y_test_original = np.expm1(y_test)

# Calculate mean squared error
mse_log = mean_squared_error(y_test, y_pred_log)
mse_original = mean_squared_error(y_test_original, y_pred_original)
print("Mean Squared Error (log scale):", mse_log)
print("Mean Squared Error (original scale):", mse_original)

# Save predictions to dataframe with titles
df_test = pd.DataFrame(X_test_scaled, columns=['budget', 'runtime', 'vote_average', 'year'] + list(genre_columns))
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