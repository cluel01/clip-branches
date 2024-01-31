from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RFSearchEngine:
    def __init__(self,rf_cfg,seed,dtype):
        self.rf_cfg = rf_cfg
        self.seed = seed
        self.dtype = dtype
        self.model = RandomForestClassifier(**rf_cfg,random_state=seed)

    def search(self,X_train, y_train,features):
        X_train = X_train.astype(self.dtype)
        self.model.fit(X_train,y_train)

        preds = self.model.predict(features)
        inds = np.where(preds == 1)[0]

        return inds




