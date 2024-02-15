import matplotlib.pyplot as plt
import numpy as np

data = [
    [1, 3],
    [2, 4],
    [3, 4],
    [5, 5],
    [6, 6],
    [7, 8]
]

x = [point[0] for point in data]
y = [point[1] for point in data]

def func(x):
    return 0.19343576800631507 * np.exp(0.4591749672382693 * x) + 3.1110373124171558

x_values = np.linspace(min(x), max(x), 100)

plt.scatter(x, y, label='data points', color='blue')
plt.plot(x_values, func(x_values), label='model output', color='red')

plt.xlabel('t Label')
plt.ylabel('y Label')
plt.legend()
plt.show()
