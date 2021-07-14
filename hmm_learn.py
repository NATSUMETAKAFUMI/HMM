#!/usr/bin/env python3

import numpy as np
from hmmlearn import hmm

model = hmm.MultinomialHMM(n_components=3)

model.startprob_ = np.array([1, 0, 0])

model.transmat_ = np.array([[0.6, 0.4, 0], 
                            [0, 0.7, 0.3], 
                            [0, 0, 1.0]])

model.emissionprob_ = np.array([[0.7, 0.2, 0.1], 
                                [0.175, 0.65, 0.175], 
                                [0.1, 0.3, 0.6]])

X, Z = model.sample(10)
print(X, Z)

X = np.array([[0], [2], [1], [1], [0], [2]])
print(model.decode(X))

X = np.array([[0], [1], [1], [2], [2], [2]])
model2 = hmm.MultinomialHMM(n_components=3).fit(X)

model2.startprob_

model2.transmat_

model2.emissionprob_

X = np.array([[0], [2], [1], [1], [0], [2]])
model2.score(X)
