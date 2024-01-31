from sklearn.tree import DecisionTreeClassifier
import numpy as np

class DTreeSearchEngine:
    def __init__(self,dtree_cfg,seed,dtype):
        self.dtree_cfg = dtree_cfg
        self.seed = seed
        self.dtype = dtype
        self.model = DecisionTreeClassifier(**dtree_cfg,random_state=seed)

    def search(self,X_train, y_train,features):
        X_train = X_train.astype(self.dtype)
        self.model.fit(X_train,y_train)

        #TODO Transform tree to range query instead
        preds = self.model.predict(features)
        inds = np.where(preds == 1)[0]

        return inds




