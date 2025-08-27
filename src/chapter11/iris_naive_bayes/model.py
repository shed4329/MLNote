import pickle
import os
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create directory for reports if it doesn't exist
report_dir = "classification_reports"
os.makedirs(report_dir, exist_ok=True)

# Load iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Print basic information about the dataset
print(f'Feature names: {iris.feature_names}')
print(f'Class names: {iris.target_names}')
print(f'Dataset shape: features={X.shape}, targets={y.shape}')

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model
print('\nEvaluation results:')
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Confusion matrix
print('\nConfusion matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Classification report
print('\nClassification report:')
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(class_report)

# Save classification report to file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"{report_dir}/iris_classification_report_{timestamp}.txt"

with open(report_filename, 'w', encoding='utf-8') as f:
    f.write("Iris Dataset - Naive Bayes Classification Report\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Feature names: {iris.feature_names}\n")
    f.write(f"Class names: {iris.target_names}\n\n")
    f.write(f"Training set size: {X_train.shape[0]} samples\n")
    f.write(f"Test set size: {X_test.shape[0]} samples\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)

print(f'\nReport saved to: {os.path.abspath(report_filename)}')

# Save the model
with open('iris_naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print('Model saved as iris_naive_bayes_model.pkl')
