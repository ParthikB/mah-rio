import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LinearRegression

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

# THRESHOLD = 50
# for THRESHOLD in range(10, 80, 10):
# max_train, max_test = pd.DataFrame(max[:-THRESHOLD]), pd.DataFrame(max[-THRESHOLD:])
# generations_train, generations_test = pd.DataFrame(generations[:-THRESHOLD]), pd.DataFrame(generations[-THRESHOLD:])
#
#     model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
#
#     # for i in types:
#     model.fit(generations_train, max_train)
#     prediction = model.predict(generations_test)
#     accuracy = accuracy_score(max_test, prediction)
#
#     print(f"Accuracy at {THRESHOLD} is {accuracy}.")

def predict(X, y):
    model = LinearRegression().fit(X, y)
    prediction = model.predict(list(range(generations[-1], generations[-1]+100)))
    # accuracy = sum((max_test/prediction)[0])/len(prediction) * 100
    return prediction