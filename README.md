# EX 5 Implementation of Logistic Regression Using Gradient Descent
## DATE:
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sreeviveka V.S
RegisterNumber:  2305001031
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,: -1]
x
y=data1.iloc[:,-1]
y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta, X,y):
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+ (1-y)* np.log(1-h))
def gradient_descent(theta, X,y, alpha, num_iterations):
  m=len(y)
for i in range(num_iterations):
  h=sigmoid(X.dot(theta))
  gradient=X.T.dot(h-y)/m
  theta-=alpha*gradient
  return theta
theta-gradient_descent(theta, X,y,alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h=sigmoid(X.dot(theta))
  y_pred-np.where(h>=0.5, 1,0)
  return y_pred

```

## Output:
![Screenshot 2024-10-17 095443](https://github.com/user-attachments/assets/f253b711-b611-465c-82d8-646ca8d2d13d)
![Screenshot 2024-10-17 095453](https://github.com/user-attachments/assets/1bb87caf-fde6-4429-aa36-ff4620d7b0ff)
![Screenshot 2024-10-17 095512](https://github.com/user-attachments/assets/5c7ae4b0-fbef-4bfc-915f-a91826c82905)
![Screenshot 2024-10-17 095519](https://github.com/user-attachments/assets/5a5e2b8a-76af-4243-ab27-ceaf72b78c88)
![Screenshot 2024-10-17 095527](https://github.com/user-attachments/assets/e3d1abb0-0b9c-44e0-a3a2-c285181dd9da)
![Screenshot 2024-10-17 095533](https://github.com/user-attachments/assets/d84d16c4-74a2-4581-abc6-68cc84506938)
![WhatsApp Image 2024-10-17 at 10 06 17_61fb0ce3](https://github.com/user-attachments/assets/29af3dc7-1510-4680-9d4f-3d62cc55365a)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

