import pandas as pd
import matplotlib.pyplot as plt

filename = 'Housing.csv'
OverflowPrevent = 1000

m = 0
b = 0
L = 0.0001
epochs = 50

GraphMin = 0
GraphMax = 200000

data = pd.read_csv("C:\\Users\\coole\\Documents\\AI\\LinearRegression\\TestData\\"+filename)

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].IV
        y = points.iloc[i].DV
        total_error += (y - (m * x + b)) ^ 2

    total_error / float(len(points))

#L = Learning Rate
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].price/OverflowPrevent
        y = points.iloc[i].lotsize/OverflowPrevent

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b



for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

print(m, b)

plt.scatter(data.price,data.lotsize)
plt.plot(list(range(GraphMin,GraphMax)), [m * x + b for x in range(GraphMin,GraphMax)], color="red")
plt.show()