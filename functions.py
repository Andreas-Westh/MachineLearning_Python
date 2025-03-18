import numpy as np
import pandas as pd
# part 1
# Generating Random Data
rg = np.random.default_rng() # generates random floats

def generate_data(n_featues, n_values):
    features = rg.random((n_featues, n_values))
    weights = rg.random((1, n_values))[0]# weights are always 1-dimensional
    targets = np.random.choice([0,1], n_featues)
    data = pd.DataFrame(features, columns=["x0","x1","x2"]) # making a df
    data["targets"] = targets
    return data, weights


def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias # see weighted sum formula in neural_network_basics
    
# our activation function (in this case sigmoid)
def sigmoid(w_sum):
    return 1 / (1+np.exp(-w_sum))

# pass prediction into loss function
def cross_entropy(target, prediction):
    return -(target*np.log10(prediction) + (1-target) * np.log10(1 - prediction))

# gradient descent step 1: update weights
def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x,w in zip(feature, weights):
        new_w = w + l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return new_weights

# gradient descent step 2: update bias
def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)
