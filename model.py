import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv('employee_attrition.csv')

df['Overtime'] = df['Overtime'].map({'Yes': 1, 'No': 0})
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

plt.xlabel('Age', size=15)
plt.ylabel('MonthlyIncome', size=15)
plt.title('Age vs MonthlyIncome', size=20)
plt.bar(df['Age'], df['MonthlyIncome'], color='purple')
plt.show()


features = ['ID', 'Age', 'DistanceFromHome', 'MonthlyIncome', 
            'JobSatisfaction', 'YearsAtCompany', 'WorkLifeBalance', 
            'Education', 'Overtime']
X = df[features]
Y = df['Attrition']


X = X.fillna(X.median())


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


input_data = pd.DataFrame([[10, 34, 9, 5000, 4, 5, 3, 3, 1]], columns=features)
prediction = model.predict(input_data)
print("Prediction for input data:", prediction)

accuracy = model.score(X_test, Y_test)
print("Accuracy:", accuracy)


y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


with open('attrition_model.pkl', 'wb') as file:
    pickle.dump(model, file)
