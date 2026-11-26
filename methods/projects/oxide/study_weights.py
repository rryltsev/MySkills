import numpy as np


def weights(rows):
    by_study = {}
    for i, row in enumerate(rows):
        by_study.setdefault(row["study"], {}).setdefault(row["sample"], []).append(i)
    result = np.zeros(len(rows))
    for recipes in by_study.values():
        for indices in recipes.values():
            result[indices] = 1/(len(by_study)*len(recipes)*len(indices))
    return result
