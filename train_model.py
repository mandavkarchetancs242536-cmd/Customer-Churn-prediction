import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Remove unnecessary column
df.drop("customerID", axis=1, inplace=True)

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ------------------ VISUALIZATIONS ------------------

# 1. Churn Count
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Count")
plt.show()

# 2. Monthly Charges vs Churn
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# 3. Tenure vs Churn
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()

# 4. Correlation Heatmap (numeric only)
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ------------------ MODEL ------------------

# Keep only required columns for simple project
X = df[["tenure", "MonthlyCharges"]]
y = df["Churn"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Save model
pickle.dump(model, open("churn_model.pkl", "wb"))

print("Model saved successfully!")
