import pandas as pd
import matplotlib.pyplot as plt

filename = 'ExampleData.csv'
OverflowPrevent = 1000

m = 0
c = -5
L = 0.0001
#epochs = 50

GraphMin = 0
GraphMax = 5

X = []
Y = []

data = pd.read_csv("C:\\Users\\coole\\Documents\\AI\\LinearRegression\\TestData\\"+filename)

def loss_function(m, c, points):
    x_error = 0
    y_error = 0
    total_x_error = 0
    total_y_error = 0
    for i in range(len(points)):
        x = points.iloc[i].IV
        y = points.iloc[i].DV

        y_error = y - (m*x+c)
        total_y_error += y_error*y_error

    X.append(c)
    Y.append(total_y_error)
    

for i in range(15):
    loss_function(m,c,data)
    c+=1



#plt.scatter(data.IV,data.DV)
#plt.plot(list(range(GraphMin,GraphMax+1)), [m * x + c for x in range(GraphMin,GraphMax+1)], color="red")

plt.scatter(X,Y)
plt.show()