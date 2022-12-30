from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris["data"][:, 3:]
y = (iris["target"]==2).astype(np.int)

# plt.scatter(x,y,color='blue')
# plt.show()

model = LogisticRegression()
model.fit(x,y)

xnew = np.linspace(0,3,1000).reshape(-1,1)
yprob = model.predict_proba(xnew)
plt.plot(xnew,yprob[:,1],"b-")
plt.show()