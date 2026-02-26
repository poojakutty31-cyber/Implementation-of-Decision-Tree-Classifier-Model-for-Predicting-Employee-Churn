# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:

1. Hardware – PCs
   
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas 

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score 

## Program:
```
import pandas as pd

# Load dataset
data = pd.read_csv("Employee.csv")

print("data.head():")
print(data.head())

print("\ndata.info():")
print(data.info())

print("\nisnull() and sum():")
print(data.isnull().sum())

print("\ndata value counts():")
print(data["left"].value_counts())


# Encode salary column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

print("\nEncoding salary column:")
data["salary"] = le.fit_transform(data["salary"])
print(data.head())


# Select Features (X) and Target (y)
print("\nSelecting Features:")
x = data[["satisfaction_level",
          "last_evaluation",
          "number_project",
          "average_montly_hours",
          "time_spend_company",
          "Work_accident",
          "promotion_last_5years",
          "salary"]]

print(x.head())

y = data["left"]


# Split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)


# Train Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", random_state=100)

dt.fit(x_train, y_train)


# Predict
y_pred = dt.predict(x_test)


# Accuracy
from sklearn import metrics
print("\nAccuracy value:")
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)


# Sample Prediction
print("\nData Prediction:")
print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]]))


# Plot Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(dt,
          feature_names=x.columns,
          class_names=["Not Left", "Left"],
          filled=True)

plt.show()
```

## Output:
<img width="1906" height="944" alt="Screenshot 2026-02-26 112407" src="https://github.com/user-attachments/assets/a3631a9a-456c-4f23-b702-48d4f94939c3" />
<img width="1882" height="951" alt="Screenshot 2026-02-26 112425" src="https://github.com/user-attachments/assets/933800b7-ad85-4e6b-ad60-ce3637985722" />
<img width="1908" height="954" alt="Screenshot 2026-02-26 112452" src="https://github.com/user-attachments/assets/b8f45c92-e301-48c7-8759-fca13bbf514e" />
<img width="1908" height="957" alt="Screenshot 2026-02-26 112508" src="https://github.com/user-attachments/assets/210c828e-4fbe-4e60-8dbe-3a772572ae36" />
<img width="1919" height="960" alt="Screenshot 2026-02-26 112530" src="https://github.com/user-attachments/assets/5541c047-cd3c-4e7e-8afd-2c2cb471ca06" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
