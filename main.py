import pandas as pd
import csv
import numpy as np
import math

df = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask1b/train.csv')


def lin_reg(k):
    # seperate train set and test set

    train_set = df.sample(frac=0.78, random_state=k)
    test_set = df.drop(train_set.index)

    # Creating numpy array phi

    phi = np.array([train_set['x1'], train_set['x2'], train_set['x3'], train_set['x4'], train_set['x5']])
    phi = np.concatenate([phi, np.square(phi), np.exp(phi), np.cos(phi)])

    phi = phi.transpose()
    phi = np.append(phi, np.ones((len(phi), 1)), axis=1)
    # phi matix is created dimesions : 420x21

    # getting the vector y

    train_y = np.array([train_set['y']])
    train_y = train_y.transpose()

    # Computing the weights in training :

    weights = np.matmul(np.linalg.inv(np.matmul(phi.transpose(), phi)), np.matmul(phi.transpose(), train_y))

    # Now that we have the weights, we use the test set :

    phi_test = np.array([test_set['x1'], test_set['x2'], test_set['x3'], test_set['x4'], test_set['x5']])
    phi_test = np.concatenate([phi_test, np.square(phi_test), np.exp(phi_test), np.cos(phi_test)])

    phi_test = phi_test.transpose()

    phi_test = np.append(phi_test, np.ones((len(phi_test), 1)), axis=1)

    # prediction of y :

    y_pred = np.matmul(phi_test, weights)
    test_y = np.array([test_set['y']])
    test_y = test_y.transpose()

    # Lastly computing the RMSE :
    n = len(y_pred)

    RMSE = math.sqrt((1 / n) * np.sum(np.square(test_y - y_pred)))
    return RMSE, weights


def optimizer(m):  # m is number of iteration
    results = np.empty(m)
    r = 0
    for i in range(len(results)):
        results[i] = lin_reg(r + i)[0]

    min_res = np.amin(results)
    results = results.tolist()
    min_indice = results.index(min_res)
    weights_opt = lin_reg(min_indice)[1]
    return weights_opt


answer = optimizer(1000)

weights_opt = pd.DataFrame(answer)
weights_opt.to_csv('/Users/lucasmichaud/Desktop/task1b/submission6')
