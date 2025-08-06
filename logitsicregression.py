import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

x = df['Age']
y = df['Income']
plt.xlabel('Age', size=15)
plt.ylabel('Income', size=15)
plt.title('Age vs Income', size=20)
plt.bar(x, y, color='red')
plt.show()

features = ['ID', 'Age', 'Experience', 'Income', 'CCAvg', 'Education', 'Mortgage', 'CD Account', 'CreditCard']
X = df[features]
Y = df['Personal Loan']
X = X.fillna(X.median())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=200)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

input_data = pd.DataFrame([[10, 34, 9, 180, 8.9, 3, 0, 0, 0]], columns=features)
prediction = model.predict(input_data)
print("The result according to the given data is", prediction)

accuracy = model.score(X_test, Y_test)
print(accuracy)

prediction = model.predict(X_test)
matrix = confusion_matrix(Y_test, prediction)
print(matrix)

print("Classification Report:\n", classification_report(Y_test, prediction))

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
