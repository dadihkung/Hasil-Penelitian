import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('fbcca_for_lda.csv')  # Replace with your actual CSV file path

# Print the first few rows to ensure it's loaded correctly
print(data.head())

# Extract features (the first 6 columns) and labels (the last column)
X = data.iloc[:, :-1].values  # Features: Frequencies (12Hz, 30Hz, 8Hz, etc.)
y = data.iloc[:, -1].values   # Labels: The actual class (12, 30, 8, etc.)

# Check the shape of the data
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the LDA model
lda = LinearDiscriminantAnalysis()

# Train the model on the training data
lda.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = lda.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Optionally, perform cross-validation to get more robust evaluation
cross_val_scores = cross_val_score(lda, X, y, cv=5)  # 5-fold cross-validation
print(f"\nCross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation accuracy: {np.mean(cross_val_scores)}")
