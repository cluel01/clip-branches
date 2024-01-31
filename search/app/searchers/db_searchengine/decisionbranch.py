import numpy as np
import time as t
from decisionbranches.models.boxSearch.boxClassifier import BoxClassifier
from decisionbranches.models.boxSearch.ensemble import Ensemble
from .multiprocessing import NoDaemonPool

class DBSearchEngine:
    def __init__(self,totfeat,indices,db_cfg,njobs,seed,dtype,min_box_size):
        self.njobs = njobs
        self.seed = seed

        #Box classifier parameters
        self.db_totfeat = totfeat
        self.indices = indices
        self.db_postTree = db_cfg.pop("postTree")
        self.min_pts = None
        if "min_pts" in db_cfg.keys():
            if db_cfg["min_pts"] == "auto":
                self.min_pts = db_cfg.pop("min_pts")
        self.db_cfg = db_cfg
        self.dtype = dtype
        self.min_box_size = min_box_size

        model = BoxClassifier(tot_feat=self.db_totfeat,indices=indices,
                                cfg=self.db_cfg,postTree=self.db_postTree,n_jobs=self.njobs,seed=self.seed,dtype=dtype)#,verbose=False)
        
        if njobs > 1:
            pool = NoDaemonPool(1,initializer=initialize_model,initargs=(model,))
            self.pool = pool
            self.model = None
        else:
            self.model = model
            self.pool = None

    def search(self,X_train,y_train,features,treeset):
        X_train = X_train.astype(self.dtype)

        if self.min_pts == "auto":
            min_pts = min(50,int(y_train.sum() * 1.5))
            self.model.init_cfg["min_pts"] = min_pts

        if self.pool is None:
            self.model.fit(X_train,y_train)

        else:
            self.model = self.pool.map(fit_model,[(X_train,y_train)])[0]

        if treeset.dtype == "uint8":
            mins,maxs,fidxs = self.model.get_boxes_uint8(min_points=self.min_box_size)
        else:
            mins,maxs,fidxs = self.model.get_boxes(min_points=self.min_box_size)

        sql_statements = self._get_sql_query(mins,maxs,fidxs)

        start = t.time()
        #### Query boxes #########
        if self.db_postTree == False:
            inds,counts,time,loaded_leaves = treeset.multi_query_ranked_cy(mins,maxs,fidxs)

        else:
            inds = []
            for idx in range(len(fidxs)):
                #TODO return not just indices but also features
                i,time,loaded_leaves = treeset.query_cy(mins[idx],maxs[idx],fidxs[idx])
                X_filter = features[i]
                tree = self.model.trees[idx]
                if tree is not None:
                    preds = self.model.trees[idx].predict(X_filter).astype(bool)
                    i = i[preds]
                inds.append(i)
            inds = np.concatenate(inds)
            inds,counts = np.unique(inds,return_counts=True)
            inds = inds[np.argsort(counts)[::-1]]
        end = t.time()
        print(f"KDtree search took: {end-start:.3f}")
        return inds,sql_statements
    
    @staticmethod
    def _get_sql_query(mins,maxs,fidxs):
        sql_statements = []

        for box_idx in range(len(mins)):
            conditions = []
            for dim_idx in range(len(fidxs[box_idx])):
                dim = fidxs[box_idx][dim_idx]
                min_val = mins[box_idx][dim_idx]
                max_val = maxs[box_idx][dim_idx]
                condition = f"(t.{dim} BETWEEN {min_val} AND {max_val})"
                conditions.append(condition)
            where_clause = " AND ".join(conditions)
            sql_statement = f"SELECT * FROM table as t WHERE {where_clause};"
            sql_statements.append(sql_statement)

        return sql_statements



class EnsembleSearchEngine:
    def __init__(self,totfeat,indices,ens_nestimators,db_cfg,njobs,seed,dtype):
        if njobs is None:
            self.njobs = ens_nestimators
        else:
            self.njobs = njobs
        self.seed = seed
        self.dtype = dtype

        #Box classifier parameters
        self.db_totfeat = totfeat
        self.indices = indices
        self.min_pts = None
        if "min_pts" in db_cfg.keys():
            if db_cfg["min_pts"] == "auto":
                self.min_pts = db_cfg.pop("min_pts")
        self.db_postTree = db_cfg.pop("postTree")
        self.db_cfg = db_cfg

        self.ens_nestimators = ens_nestimators

        self.model = Ensemble(n_estimators=self.ens_nestimators,tot_feat=self.db_totfeat,indices=indices,
                        cfg=self.db_cfg,postTree=self.db_postTree,n_jobs=self.njobs,seed=self.seed,dtype=self.dtype)#,verbose=False)

    def search(self,X_train, y_train,features,treeset):
        X_train = X_train.astype(self.dtype)
        # #TODO do this filtering in the fronted when selecting the cell to save query time
        # idxs_rare = np.intersect1d(idxs_rare,self.mapping[:,0])
        # idxs_nonrare = np.intersect1d(idxs_nonrare,self.mapping[:,0])

        if self.min_pts == "auto":
            min_pts = min(50,int(y_train.sum() * 1.5))
            self.model.cfg["min_pts"] = min_pts

        self.model.fit(X_train,y_train)

        if treeset.dtype == "uint8":
            mins,maxs,fidxs = self.model.get_boxes_uint8()
        else:
            mins,maxs,fidxs = self.model.get_boxes()

        start = t.time()
        #### Query boxes #########
        inds,counts,time,loaded_leaves = treeset.multi_query_ranked_cy(mins,maxs,fidxs)

        inds = inds[np.where(counts > self.ens_nestimators // 2)]
        end = t.time()
        print(f"KDtree search took: {end-start:.3f}")

        return inds

    
def initialize_model(m):
    global model
    model = m

def fit_model(args):
    X_train,y_train = args
    model.fit(X_train,y_train)
    return model