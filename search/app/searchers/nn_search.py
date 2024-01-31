import numpy as np
from decisionbranches.models.boxSearch.boxClassifier import BoxClassifier
from decisionbranches.models.boxSearch.ensemble import Ensemble


class NNSearchEngine:
    def __init__(self,k,seed,dtype):
        self.k = k
        self.seed = seed
        self.dtype = dtype

    def search(self,X_train,y_train,treeset):
        np.random.seed(self.seed)

        trees = list(treeset.trees.keys())

        random_idx = np.random.choice(np.arange(len(trees)),size=1)[0]
        t_key = trees[random_idx]

        kdtree = treeset.trees[t_key]
        feat_idxs = treeset.indexes[random_idx]

        rare_idx = np.random.choice(np.where(y_train == 1)[0],size=1)
        X_train = X_train[rare_idx,feat_idxs]

        inds,dist,_,_ = kdtree.query_point_cy(X_train,k=self.k)

        return inds