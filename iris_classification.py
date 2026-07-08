# ============================================================
# UoPeople - Data Mining and Machine Learning
# Programming Assignment - Mohpheth Ekhaguere
# Classification Models for Flower Species Prediction
# Part 1
# ============================================================

# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING IRIS DATASET")
print("=" * 60)

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["Species"] = y

print("\nFirst Five Rows")
print(df.head())

# ============================================================
# DATASET CHARACTERISTICS
# ============================================================

print("\n" + "=" * 60)
print("DATASET CHARACTERISTICS")
print("=" * 60)

print(f"\nNumber of Samples : {len(df)}")

print(f"\nNumber of Features : {len(feature_names)}")

print("\nFeature Names")

for feature in feature_names:
    print("-", feature)

print("\nTarget Classes")

for i, name in enumerate(target_names):
    print(f"{i} : {name}")

print("\nClass Distribution")

print(df["Species"].value_counts())

print("\nDataset Information")

print(df.info())

print("\nSummary Statistics")

print(df.describe())

# ============================================================
# SPLIT DATASET
# ============================================================

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train)}")

print(f"Testing Samples  : {len(X_test)}")

# ============================================================
# SIMPLE RULE BASED CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("RULE-BASED CLASSIFIER")
print("=" * 60)

# Rule:
# If petal length < 2.5 -> Setosa
# Else if petal width < 1.8 -> Versicolor
# Else -> Virginica

def rule_classifier(sample):

    petal_length = sample[2]
    petal_width = sample[3]

    if petal_length < 2.5:
        return 0

    elif petal_width < 1.8:
        return 1

    else:
        return 2


rule_predictions = []

for sample in X_test:
    prediction = rule_classifier(sample)
    rule_predictions.append(prediction)

rule_predictions = np.array(rule_predictions)

print("\nRule-Based Classifier Completed")

print("\nSample Predictions")

for i in range(10):

    print(
        f"Actual : {target_names[y_test[i]]:<12}"
        f" Predicted : {target_names[rule_predictions[i]]}"
    )

print("\nRule-Based Accuracy")

print(f"{accuracy_score(y_test, rule_predictions):.4f}")

print("\nConfusion Matrix")

print(confusion_matrix(y_test, rule_predictions))

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rule_predictions,
        target_names=target_names
    )
)

print("\nPart 1 Completed Successfully.")
# ============================================================
# PART 2
# BUILD AND TEST CLASSIFICATION MODELS
# ============================================================

print("\n" + "=" * 60)
print("BUILDING CLASSIFICATION MODELS")
print("=" * 60)

# Dictionary to store models
models = {}

# Dictionary to store predictions
predictions = {}

# ------------------------------------------------------------
# DECISION TREE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DECISION TREE")
print("=" * 60)

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)

models["Decision Tree"] = decision_tree
predictions["Decision Tree"] = dt_predictions

print("Decision Tree model trained successfully.")

print("\nFirst 10 Predictions")

for i in range(10):
    print(
        f"Actual: {target_names[y_test[i]]:<12}"
        f" Predicted: {target_names[dt_predictions[i]]}"
    )


# ------------------------------------------------------------
# NAIVE BAYES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NAIVE BAYES")
print("=" * 60)

naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)

nb_predictions = naive_bayes.predict(X_test)

models["Naive Bayes"] = naive_bayes
predictions["Naive Bayes"] = nb_predictions

print("Naive Bayes model trained successfully.")

print("\nFirst 10 Predictions")

for i in range(10):
    print(
        f"Actual: {target_names[y_test[i]]:<12}"
        f" Predicted: {target_names[nb_predictions[i]]}"
    )


# ------------------------------------------------------------
# LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logistic = LogisticRegression(
    max_iter=500,
    random_state=42
)

logistic.fit(X_train, y_train)

log_predictions = logistic.predict(X_test)

models["Logistic Regression"] = logistic
predictions["Logistic Regression"] = log_predictions

print("Logistic Regression model trained successfully.")

print("\nFirst 10 Predictions")

for i in range(10):
    print(
        f"Actual: {target_names[y_test[i]]:<12}"
        f" Predicted: {target_names[log_predictions[i]]}"
    )


# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("K-NEAREST NEIGHBORS")
print("=" * 60)

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

models["KNN"] = knn
predictions["KNN"] = knn_predictions

print("KNN model trained successfully.")

print("\nFirst 10 Predictions")

for i in range(10):
    print(
        f"Actual: {target_names[y_test[i]]:<12}"
        f" Predicted: {target_names[knn_predictions[i]]}"
    )


# ------------------------------------------------------------
# SUPPORT VECTOR MACHINE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SUPPORT VECTOR MACHINE")
print("=" * 60)

svm = SVC(
    kernel="linear",
    random_state=42
)

svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

models["SVM"] = svm
predictions["SVM"] = svm_predictions

print("SVM model trained successfully.")

print("\nFirst 10 Predictions")

for i in range(10):
    print(
        f"Actual: {target_names[y_test[i]]:<12}"
        f" Predicted: {target_names[svm_predictions[i]]}"
    )


print("\n" + "=" * 60)
print("ALL FIVE MACHINE LEARNING MODELS HAVE BEEN TRAINED")
print("=" * 60)

print("""
Models Completed

1. Rule-Based Classifier
2. Decision Tree
3. Naive Bayes
4. Logistic Regression
5. K-Nearest Neighbors
6. Support Vector Machine
""")

print("Part 2 Completed Successfully.")
# ============================================================
# PART 3
# MODEL EVALUATION, COMPARISON AND ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

results = []

# Include the rule-based classifier
all_predictions = {
    "Rule-Based": rule_predictions,
    "Decision Tree": dt_predictions,
    "Naive Bayes": nb_predictions,
    "Logistic Regression": log_predictions,
    "KNN": knn_predictions,
    "SVM": svm_predictions
}

for model_name, prediction in all_predictions.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(y_test, prediction)

    print("\nAccuracy")
    print(f"{accuracy:.4f}")

    print("\nPrecision")
    print(f"{precision:.4f}")

    print("\nRecall")
    print(f"{recall:.4f}")

    print("\nF1 Score")
    print(f"{f1:.4f}")

    print("\nConfusion Matrix")
    print(matrix)

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            prediction,
            target_names=target_names,
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print(results_df)

# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST PERFORMING MODEL")
print("=" * 60)

print(best_model)

# ============================================================
# ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

print("""
1. Models were compared using:
   - Confusion Matrix
   - Precision
   - Recall
   - F1 Score

2. The best model is the one with the highest F1 Score.

3. The Iris dataset is relatively easy to classify because:
   - It has only four numerical features.
   - The three flower species are well separated.
   - The dataset is balanced with 50 samples per class.

4. Because the classes are well separated,
   most supervised learning algorithms perform
   very well on this dataset.

5. Simpler models such as Logistic Regression
   and Naive Bayes can achieve excellent results,
   while Decision Tree, KNN and SVM may achieve
   even higher performance depending on the
   training/testing split.

6. The Rule-Based classifier is included to
   demonstrate how manually written rules can
   classify data. However, it is generally less
   flexible than machine learning algorithms.
""")

print("\n" + "=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# VISUALIZATION 1
# MODEL COMPARISON
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["F1 Score"]
)

plt.title("Comparison of Classification Models")
plt.xlabel("Models")
plt.ylabel("F1 Score")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()



# ============================================================
# VISUALIZATION 2
# CONFUSION MATRIX FOR BEST MODEL (SVM)
# ============================================================

cm = confusion_matrix(y_test, svm_predictions)

plt.figure(figsize=(6,5))

plt.imshow(cm)

plt.title("Confusion Matrix - Support Vector Machine")

plt.colorbar()

plt.xticks(
    [0,1,2],
    target_names
)

plt.yticks(
    [0,1,2],
    target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i,j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()