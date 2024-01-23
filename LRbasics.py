import pandas as pd
import matplotlib.pyplot as plt

filename = 'ExampleData.csv'
OverflowPrevent = 1
UpdateInterval = 50

m = 0
c = 0
L = 0.0001
epochs = 1000

GraphMin = 0
GraphMax = 5

data = pd.read_csv("C:\\Users\\coole\\Documents\\AI\\LinearRegression\\TestData\\"+filename)

#L = Learning Rate
def gradient_descent(m_now, c_now, points, L):
    dedm = 0
    dedc = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].IV/OverflowPrevent
        y = points.iloc[i].DV/OverflowPrevent

        dedm += -(2/n) * x * (y - (m_now * x + c_now))
        dedc += -(2/n) * (y - (m_now * x + c_now))

    m = m_now - dedm * L
    c = c_now - dedc * L

    return m,c

for i in range(epochs):
    if i % UpdateInterval == 0:
        print(f'Epoch: {i}')
    m, c = gradient_descent(m, c, data, L)

print(m, c)

plt.scatter(data.IV,data.DV)
plt.plot(list(range(GraphMin,GraphMax+1)), [m * x + c for x in range(GraphMin,GraphMax+1)], color="red")

plt.show()