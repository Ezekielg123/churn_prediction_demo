import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
data = {
    "Age": [25, 45, 35, 50, 23, 40, 60, 48, 33, 29],
    "UsageHours": [40, 10, 20, 5, 45, 15, 3, 8, 25, 38],
    "MonthlyBill": [300, 500, 400, 550, 280, 480, 600, 520, 420, 310],
    "Churn": [0, 1, 0, 1, 0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
print(df)

X=df[["Age","UsageHours","MonthlyBill"]]
Y=df["Churn"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression()

model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

c=confusion_matrix(Y_test,Y_pred)
accuracy=accuracy_score(Y_test,Y_pred)

plt.figure()
plt.imshow(c)
plt.title("Confusion matix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()