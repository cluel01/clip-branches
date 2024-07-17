import numpy as np
import pandas as pd
import torch
import math

import time
import os
import multiprocessing as mp

from ...cython.functions import filter_py

from .boxSearch import BoxSearch
from ...utils import helpers

'''
Class for DecisionBranch Ensemble
'''
class Ensemble():
    def __init__(self,tot_feat,n_estimators=10,n_feat=5,n_ind=1000,max_feat="all",indices=None,n_jobs=5,max_samples=None,bootstrap=True,
                    cfg=None,feats=None,postTree=False, verbose=True,debug=False,dtype="float64",dtype_int = "int32",seed=42):
        '''
        tot_feat: total number of features
        n_estimators: number of estimators within the ensemble
        n_feat: number of features per subset
        n_ind: number of feature subsets
        max_feat: number of feature subsets that are evaluated per iteration 
        indices: predefined feature subsets
        n_jobs: number of cores to use
        max_samples: relevant when bootstrap is on (size of bootstrap samples)
        bootstrap: en-/disable bootstrap
        cfg: config dictionary for specifiy config parameters of single decision branches 
        feats: predefined number of features to be used for subset generation
        postTree: if top-down construction on all features is en-/disabled
        verbose: verbose output
        debug: debug output
        dtype: float dtype
        dtype_int: int dtype
        seed: random seed
        '''
        np.random.seed(seed)

        if indices is not None:
            if isinstance(indices,list):
                indices = np.array(indices,dtype=dtype_int)
            self.feat_idxs = indices
            n_ind,n_feat = indices.shape
        else:
            if feats is not None:
                feats = feats
            elif (feats is None) & (max_feat != "all"):
                feats = np.random.choice(tot_feat, size=max_feat, replace=False)
            else:
                feats = np.arange(tot_feat,dtype=dtype_int)

            if n_feat > tot_feat:
                n_feat = tot_feat

            if n_ind == "all":
                n_ind = math.comb(len(feats),n_feat)
                print(f"All {n_ind} possible combinations are evaluated!")
            else:
                if n_ind > math.comb(len(feats),n_feat):
                    n_ind = math.comb(len(feats),n_feat)

            #Build feature indices
            feat_idxs = helpers.generate_fidxs(n_feat,n_ind,feats,dtype_int,seed)
            self.feat_idxs = feat_idxs

        if n_jobs > os.cpu_count():
            print("Info: n_jobs is larger than available CPUs! Set to n_cpus.")
            n_jobs = os.cpu_count()
 
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.cfg = cfg
        self.n_jobs = n_jobs
        self.debug = debug
        self.verbose = verbose
        self.postTree = postTree
         
        self.feats = feats
        self.dtype = dtype
        self.dtype_int = dtype_int
        self.seed = seed
        self.max_feat = max_feat
        self.tot_feat = tot_feat
        self.n_feat = n_feat
        self.n_ind = n_ind

        self.estimators = None

    '''
    Function to train models in ensemble
    '''
    def fit(self,x,y,pos_label=1):
        if isinstance(x,pd.DataFrame):
            x = np.array(x,dtype=self.dtype)
        elif isinstance(x,torch.Tensor):
            x = x.numpy().astype(self.dtype)
        elif isinstance(x,np.ndarray):
            if x.dtype != self.dtype:
                x = x.astype(self.dtype)
        else:
            raise ValueError("No known data type for x!")

        if y.dtype != np.dtype(self.dtype_int):
            y = y.astype(self.dtype_int)

        start = time.time()

        estimators = []
        masks = []
        seed = self.seed
        for i in range(self.n_estimators):
            seed = np.random.randint(100000000)
            se = BoxSearch(verbose=self.debug,debug=self.debug,dtype=self.dtype,seed=seed)
            se.set_feat_idxs(self.feat_idxs,self.tot_feat)
            if self.bootstrap:
                masks.append(self._bootstrap(x,y,seed))
            else:
                masks.append([])
            estimators.append(se)

        #pool = mp.Pool(self.n_jobs,initializer=self._init_worker,initargs=(x,y,pos_label),maxtasksperchild=None)
        pool = mp.Pool(self.n_jobs,initializer=self._init_worker,initargs=(x,y,pos_label),maxtasksperchild=None)
        #pool = ThreadPool(self.n_jobs)
        try:
            self.estimators = pool.map(self._worker, ((se, self.n_jobs, self.cfg,masks[i], self.postTree) for i,se in enumerate(estimators)))
        except Exception as e:
            print(e)
            pool.close()
            pool.join()
        end = time.time()
        pool.close()
        pool.join()
        if self.verbose:
            print("Number of covered features: ",len(np.unique(self.feat_idxs)))
            print("Training time: ",end-start)

    '''
    Function for making binary classification prediction given data x
    '''
    def predict(self,x,threshold = 0.5):
        pred = np.zeros(len(x),dtype=self.dtype_int)
        probs = self.predict_proba(x)
        pred[probs > threshold] = 1
        return pred
    
    '''
    Function for predicting the class probabilities given data x
    '''
    def predict_proba(self,x):
        count = np.zeros(len(x),dtype=self.dtype_int)
        for i in range(self.n_estimators):
            pred = self._predict(x,i)
            count = count + pred
        probs = count/self.n_estimators
        return probs

    def _predict(self,x,eid): 
        if isinstance(x,pd.DataFrame):
            x = np.array(x,dtype=self.dtype)
        elif isinstance(x,torch.Tensor):
            x = x.numpy().astype(self.dtype)
        elif isinstance(x,np.ndarray):
            if x.dtype != self.dtype:
                x = x.astype(self.dtype)
        else:
            raise ValueError("No known data type for x!")
        pos_idx = self._search(x,eid)
        mask = np.zeros(len(x),dtype=int)
        if len(pos_idx) > 0:
            mask[pos_idx] = 1  
        return mask           


    def _search(self,x,eid):
        if isinstance(x,pd.DataFrame):
            x = np.array(x,dtype=self.dtype)
        elif isinstance(x,torch.Tensor):
            x = x.numpy().astype(self.dtype)
        elif isinstance(x,np.ndarray):
            if x.dtype != self.dtype:
                x = x.astype(self.dtype)
        else:
            raise ValueError("No known data type for x!")

        assert x.shape[1] == self.tot_feat, "Test data must have the same dimensionality as original data!" 

        bboxes = self.estimators[eid]

        pts = []
        mask = np.ones(len(x),dtype=bool)
        if not self.postTree:
            n_boxes = len(bboxes)
        else:
            n_boxes = len(bboxes[0])

        for i in range(n_boxes):
            if not self.postTree:
                box,feat_idxs = bboxes[i]
            else:
                box,feat_idxs = bboxes[0][i]
            output = np.ones(np.sum(mask),dtype=np.intc)
            filter_py(box,x[mask][:,feat_idxs],output)
            output = np.asarray(output,dtype=bool)
            if np.sum(output) > 0:
                is_empty = False
                if self.postTree:
                    tree = bboxes[1][i]
                    if tree is not None:
                        preds = tree.predict(x[mask][output]).astype(bool)
                        pts_idx = np.arange(len(x))[mask][output][preds]
                    else:
                        is_empty = True
                if (self.postTree == False) or (is_empty):
                    pts_idx = np.arange(len(x))[mask][output]
                mask[pts_idx] = False
                pts.extend(pts_idx)
        return np.array(pts)

    def _bootstrap(self,x,y,seed):
        np.random.seed(seed)
        if isinstance(self.max_samples,int): 
            sample_size = self.max_samples
        elif isinstance(self.max_samples,float):
            sample_size = self.max_samples * len(x)
        else:
            sample_size = len(x)
            
        while True:
            sample_idx = np.random.choice(len(x), size=sample_size, replace=True)
            if np.sum(y[sample_idx]) < 1:
                continue
            return sample_idx
        
    '''
    Function to output the found boxes during training
    '''
    def get_boxes(self):
        n_boxes = 0
        for j in self.estimators:
            n_boxes += len(j)

        mins = np.empty((n_boxes,self.n_feat),dtype=self.dtype)
        maxs = np.empty((n_boxes,self.n_feat),dtype=self.dtype)
        fidxs = np.empty((n_boxes,self.n_feat),dtype=self.dtype_int)

        c = 0
        for e in self.estimators:

            for j in range(len(e)):
                box,f_idx = e[j]
                mins[c] = box[:,0]
                maxs[c] = box[:,1]
                fidxs[c] = f_idx
                c += 1
        fidxs = fidxs.tolist()
        return mins,maxs,fidxs
           

    @staticmethod
    def _init_worker(x,y,Pos_label):
        global x_glob,y_glob,pos_label

        x_glob,y_glob,pos_label = x,y,Pos_label


    @staticmethod
    def _worker(args):
        se,  n_jobs, cfg,mask, postTree = args

        if len(mask) > 0:
            x = x_glob[mask]
            y = y_glob[mask]
            se.load_data(x,y,pos_label=pos_label)
        else:
            se.load_data(x_glob,y_glob,pos_label=pos_label)
        if postTree:
            se.train_boxSearch(**cfg)
            se.train_postTree()
            return (se.bboxes,se.trees)
        else:
            se.train_boxSearch(**cfg)
            return se.bboxes

    
