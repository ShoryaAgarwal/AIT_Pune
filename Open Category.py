import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
chess_data = pd.read_csv('chess train.csv')

# Preprocessing
# Handling missing values if any
chess_data.dropna(inplace=True)

# Encode categorical variables using one-hot encoding
chess_data = pd.get_dummies(chess_data, columns=['opening_name', 'rated'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['white_rating', 'black_rating', 'turns']
chess_data[numerical_features] = scaler.fit_transform(chess_data[numerical_features])

# Extract features and target variable
X = chess_data.drop('winner', axis=1)
y = chess_data['winner']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Model evaluation with cross-validation
cv_accuracy = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_accuracy.mean())

# Final model evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))