"""
Movie Rating Prediction using Regression

This script predicts IMDb movie ratings based on movie features
such as runtime, votes, revenue, and metascore.
"""

# ---------------------------
# 1. Import Required Libraries
# ---------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# 2. Load Dataset
# ---------------------------
DATA_PATH = r"C:\Users\lenovo\PycharmProjects\movie_rating_prediction\IMDB-Movie-Data.csv"

df = pd.read_csv(DATA_PATH)

# ---------------------------
# 3. Select Relevant Features
# ---------------------------
# These features strongly influence movie ratings
features = [
    'Runtime (Minutes)',
    'Votes',
    'Revenue (Millions)',
    'Metascore'
]

target = 'Rating'

# Keep only required columns
df = df[features + [target]]

# ---------------------------
# 4. Handle Missing Values
# ---------------------------
# Replace missing values with column mean
df.fillna(df.mean(), inplace=True)

# ---------------------------
# 5. Split Features and Target
# ---------------------------
X = df[features]
y = df[target]

# ---------------------------
# 6. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 7. Feature Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 8. Train Regression Model
# ---------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------
# 9. Make Predictions
# ---------------------------
y_pred = model.predict(X_test_scaled)

# ---------------------------
# 10. Evaluate Model Performance
# ---------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# ---------------------------
# 11. Predict Rating for a New Movie
# ---------------------------
# Example: unseen movie
new_movie = pd.DataFrame({
    'Runtime (Minutes)': [120],
    'Votes': [150000],
    'Revenue (Millions)': [80],
    'Metascore': [70]
})

new_movie_scaled = scaler.transform(new_movie)
predicted_rating = model.predict(new_movie_scaled)

print("\nPredicted IMDb Rating for New Movie:")
print(f"{predicted_rating[0]:.1f}")
