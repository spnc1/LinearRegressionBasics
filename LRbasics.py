import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\coole\\Documents\\AI\\LinearRegression\\TestData\\Housing.csv")

plt.scatter(data.price, data.lotsize)
plt.show()