import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Loading and splitting the data
train_data = pd.read_csv('datasets/train.csv')

# Separate features and target variables
X = train_data.drop(columns=['A', 'B', 'C', 'D', 'E', 'Y'])  # Features (wavelengths 3800-7377)
y = train_data[['A', 'B', 'C', 'D', 'E', 'Y']]  # Targets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Train model
models = {}
predictions = {}

for target in ['A', 'B', 'C', 'D', 'E', 'Y']:
    print(f"Training model for {target}...")
    
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Fit the model with the best hyperparameters
    grid_search.fit(X_train_scaled, y_train[target])
    
    # Get the best model
    best_model = grid_search.best_estimator_
    models[target] = best_model
    
    # Predicting
    pred = best_model.predict(X_val_scaled)
    predictions[target] = pred
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_val[target], pred)
    print(f"MAPE for {target}: {mape:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val[target], pred, alpha=0.6, label='Predictions')
    plt.plot([y_val[target].min(), y_val[target].max()], [y_val[target].min(), y_val[target].max()], 'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted Values for {target}')
    plt.legend()
    plt.show()

# Output
prediction_df = pd.DataFrame(predictions)
print(prediction_df)
