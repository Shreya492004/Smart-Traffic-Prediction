import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import joblib

# Load dataset
data = pd.read_csv("data/traffic_data.csv")

# Convert timestamp
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Encode DayOfWeek
le = LabelEncoder()
data['DayOfWeek'] = le.fit_transform(data['DayOfWeek'])

# Map TrafficLevel
traffic_mapping = {'Low':0, 'Normal':1, 'High':2, 'Heavy':3}
data['TrafficLevel'] = data['TrafficLevel'].map(traffic_mapping)

# Features
X = data[['DayOfWeek','CarCount','BikeCount','BusCount','TruckCount','TotalVehicleCount']]
y = data['TrafficLevel']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# Model
model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Save everything
joblib.dump(model, "xgboost_traffic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model trained and saved successfully!")
