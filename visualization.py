import numpy as np
import matplotlib.pyplot as plt
from handlers import *

data = list(np.load(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}/fitness_log.npy"))

mean = []
max = []
rep = []
generations = []

for x in data:
    mean.append(x[0])
    max.append(x[1])
    rep.append(x[2])

for i in range(len(max)):
    generations.append(i)

fig, ax1 = plt.subplots()
ax1.grid()
ax1.set_xlabel("Generations")
ax1.set_ylabel("Fitness")
ax1.plot(generations, mean, '#3333ff', label='mean')
ax1.plot(generations, max, '#FF8C00', label='max')
ax1.tick_params(axis='y')
ax1.yaxis.label.set_color((0, 0, 0.4, 1))
plt.legend()



ax2 = ax1.twinx()
ax2.set(ylim=(0, 10))
ax2.set_ylabel("Times it reached Max Fitness")
ax2.yaxis.label.set_color((0.25, 0.5, 0, 1))
ax2.plot(generations, rep, color='#408000', alpha=0.8)
ax2.tick_params(axis='y')
ax2.fill_between(generations, 0, rep, facecolor='g', alpha=0.5)

fig.tight_layout()
plt.show()