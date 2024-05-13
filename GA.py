from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd 

from genetic_selection import GeneticSelectionCV


def main():
    train = pd.read_csv('heart_dataset/dataset')
    test = pd.read_csv('heart_dataset/test.txt')
    test_X = test.values[:, 0:12] 
    X = train.values[:, 0:12] 
    y = train.values[:, 13]

    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=10,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=200,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print(selector.support_)
    y_pred = selector.predict(test_X) 
    print(y_pred)

if __name__ == "__main__":
    main()
