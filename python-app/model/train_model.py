import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('../data/logistics_data.csv')

# Features & target
X = df[['volume_cft', 'weight_kg', 'distance_km']]
y = df['optimal_vehicle']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(f'Model Accuracy: {model.score(X_test, y_test):.2f}')

# Save the trained model
joblib.dump(model, 'vehicle_selector_model.pkl')
