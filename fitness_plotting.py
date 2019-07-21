import numpy as np
import matplotlib.pyplot as plt


log = list(np.load("best_fitness_log.npy"))
y = []
x = []

for i in log:
    y.append(i[0])

for i in range(len(y)):
    x.append(i+1)

xa = np.array(x)
ya = np.array(y)

plt.plot(x, y)
plt.ylabel("Fitness")
plt.xlabel("Generation")
plt.title("Fitness vs Generation")
plt.show()