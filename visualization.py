import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from handlers import *


def return_list(x):
    list = []
    for i in x:
        list.append(i[0])
    return list


def predict(X, y, generations):
    X, y = pd.DataFrame(X), pd.DataFrame(y)
    model = LinearRegression().fit(X, y)
    future_generations = (np.array(list(range(gen[-1], gen[-1] + generations)))).reshape(-1, 1)
    prediction = model.predict(future_generations)
    # accuracy = sum((max_test/prediction)[0])/len(prediction) * 100
    return return_list(prediction), list(range(gen[-1], gen[-1] + generations))


def plot(prediction=True, generations=50):
    # Plotting
    fig, ax1 = plt.subplots()

    ax_prediction = ax_fitness = ax1.twinx()
    ax1.set(ylim=(0, 10))
    plt.title(f"Version {VERSION}")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Times it reached Max Fitness")
    ax1.yaxis.label.set_color((0.25, 0.5, 0, 1))
    ax1.plot(gen, rep, color='#408000', alpha=0.3)
    ax1.fill_between(gen, 0, rep, facecolor='#408000', alpha=0.15)
    ax1.tick_params(axis='y')

    ax_fitness.grid()
    ax_fitness.set_ylabel("Fitness")
    ax_fitness.plot(gen, mean, '#3333ff', label='Mean')
    ax_fitness.plot(gen, max, '#FF8C00', label='Max')
    ax_fitness.tick_params(axis='y')
    ax_fitness.yaxis.label.set_color((0, 0, 0.4, 1))
    # plt.legend()


    predicted_max, future_generations = predict(gen, max, generations=generations)
    predicted_mean, future_generations = predict(gen, mean, generations=generations)

    if prediction:
        ax_prediction.plot(future_generations, predicted_mean, '--b', label='Predicted mean', alpha=0.5)
        ax_prediction.plot(future_generations, predicted_max, '--r', label='Predicted max', alpha=0.3)

    plt.legend(loc=2)


    fig.tight_layout()
    plt.show()

##########################################################################


max, mean, gen, rep = data_loader(division=25)

plot(prediction=False, generations=100)

