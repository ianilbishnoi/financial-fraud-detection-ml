import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("creditcard.csv")

print("Dataset Loaded")
print(data.head())

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)

print("Training model...")
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

print("Model Evaluation:")
print(classification_report(y_test, pred))

# Confusion matrix
cm = confusion_matrix(y_test, pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Credit Card Fraud Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()