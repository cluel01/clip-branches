import numpy as np
#Only required for checking the input
import pandas as pd
import torch 

from scipy.special import comb
import os 
import time 

import multiprocessing as mp
from ...utils import helpers 
from ...cython.functions import filter_py,search_boxes
from sklearn.tree import DecisionTreeClassifier

split_types = {"half":0, "max":1, "random":2, "min":3}


'''
Class BoxSearch describes the DecisionBranch construction phase to find meaningful hyperrectangles 
among the given feature subsets. Here, the term "box" can be understand as one hyperrectangle. 
'''
class BoxSearch(object):
    def __init__(self, verbose=True,debug=False,dtype="float64",dtype_int="int32",seed=42):
        """
            verbose: verbose output
            debug: debug output
            dtype: float dtype
            dtype_int: dtype for int
            seed: random seed
        """
        self.seed = seed
        np.random.seed(seed)
        
        self.verbose = verbose
        self.debug = debug
        self.dtype=dtype
        self.dtype_int = dtype_int
        self.postTree = False

    '''
    Function to randomly generate feature subsets with the given characteristics in case no 
    feature subsets are given.
    '''
    def build_indices(self,n_feat,n_ind,tot_feat,max_feat="all",feats=None):
        '''
            n_feat: number of features per subset
            n_ind: number of feature subsets
            tot_feat: total number of available features
            max_feat: limit for number of features to be used for feature subsets
            feats: predefine the number of used features to use
        '''
        np.random.seed(self.seed)
        if n_feat > tot_feat:
            if self.verbose:
                print("Warning: n_feat needs to be equal or preferably smaller than the number of total features!")
            n_feat = tot_feat

        if feats is None:
            if max_feat ==  "all":
                max_feat = tot_feat
                feats = np.arange(tot_feat)
            else:
                if max_feat > tot_feat:
                    max_feat = tot_feat
                    feats = np.arange(tot_feat)
                else:
                    feats = np.random.choice(np.arange(tot_feat), size=max_feat, replace=False)

        if n_ind == "all":
            n_ind = comb(len(feats),n_feat)
            print(f"All {n_ind} possible combinations are evaluated!")
        else:
            if n_ind > comb(len(feats),n_feat):
                n_ind = comb(len(feats),n_feat)
        
        if n_feat < tot_feat:
            feat_idxs = helpers.generate_fidxs(n_feat,n_ind,feats,dtype=self.dtype_int,seed=self.seed)
        elif n_feat == tot_feat:
            n_ind = 1
            feat_idxs = feats[np.newaxis,...]

        self.feat_idxs = feat_idxs.astype(self.dtype_int)
        self.n_feat = n_feat
        self.n_ind = n_ind
        self.tot_feat = tot_feat
        self.feats = feats

    '''
        Set the feature subsets manually
    '''
    def set_feat_idxs(self,feat_idxs,tot_feat):
        '''
        feat_idxs: array/list of feature subsets
        tot_feat: total number of features
        '''
        if isinstance(feat_idxs,list):
            feat_idxs = np.array(feat_idxs,dtype=self.dtype_int)
        elif isinstance(feat_idxs,np.ndarray):
            if feat_idxs.dtype != np.dtype(self.dtype_int):
                feat_idxs = feat_idxs.astype(self.dtype_int)

        self.feat_idxs = feat_idxs
        self.n_ind, self.n_feat = feat_idxs.shape
        self.tot_feat = tot_feat
        self.feats = None


    """
    Function to load the training data into memory
    Input: positive and negative samples of data by index of each sample
    """
    def load_data(self, x,y,pos_label=1):
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

        if pos_label != 1:
            mask = (y==pos_label)
            self.y = y.copy()
            #binarize the labels
            self.y[mask] = 1
            self.y = y[~mask] = 0
        else:
            self.y = y

        if self.tot_feat == 0:
            self.tot_feat = x.shape[1]

        self.x = x

    def train_boxSearch(self, max_evals="auto",max_nbox=100,min_nobj=1, splitter="half",min_pts=5,
                    early_stopping=True,class_weights=[1,1],del_nonrare=True,rand_order=True,stop_infinite_split="random",
                    stop_infinite=False,top_down=False,eps=1e-16,retry_limit=5):
        '''
            max_evals: number of evaluations among all available feature subsets 
            max_nbox: maximum number of boxes 
            min_nobj:  minimum number of points in box
            splitter: splitting method to split (choose between half, min, max, random)
            min_pts: number of points to compare in each direction for expanding the boundary
            early_stopping : en-/disable early stopping (when no rare point exists in one direction during expansion)
            class_weight: class weights for both classes 
            del_nonrare: en-/disable if non-rare points should be removed when contained in box
            rand_order: en-/disable random order processing of the features within feature subset
            stop_infinite: en-/disable expansion until infinity
            top_down: en-/disable top-down construction (to split impure box into multiple pure boxes)
            eps: epsilon factor to prevent numerical instability 
            retry_limit: number of retries when no box has been found that has better impurity than -infinity
        '''
        n_jobs = 1
        np.random.seed(self.seed)

        self.max_evals_input = max_evals

        #Random feature selection equal to Random Forest
        if max_evals == "auto":
            max_evals = int(np.ceil(np.sqrt(len(self.feat_idxs))))
        elif max_evals == "log2":
            max_evals = int(np.ceil(np.log2(len(self.feat_idxs))))
        elif type(max_evals) == float:
            max_evals = int(np.ceil(len(self.feat_idxs) * max_evals))
        elif max_evals == "all":
            max_evals = len(self.feat_idxs)
        if max_evals > len(self.feat_idxs):
            max_evals = len(self.feat_idxs)

        if min_pts > len(self.x):
            min_pts = len(self.x)

        #in case only positive class weight is given
        if isinstance(class_weights,int):
            class_weights = [1,class_weights]
        
        self.cfg = {"n_jobs": n_jobs, "max_nbox": max_nbox, "min_nobj": min_nobj,
                    "max_evals": max_evals,"min_pts":min_pts,"early_stopping":early_stopping,
                    "splitter":splitter,"class_weights": class_weights,"del_nonrare":del_nonrare,
                   "rand_order": rand_order,"stop_infinite":stop_infinite,"stop_infinite_split":stop_infinite_split,"top_down":top_down,
                   "dtype":self.dtype,"eps":eps,"seed": self.seed}

        #Init for Cython functions
        early_stopping = int(early_stopping)
        stop_infinite = int(stop_infinite)
        stop_infinite_split = split_types[stop_infinite_split]
        rand_order = int(rand_order)
        splitter = split_types[splitter]
        class_weights = np.asarray(class_weights,dtype=self.dtype_int)
        box = np.empty((self.n_feat,2),dtype=self.dtype)
        pts_idx = np.empty(len(self.x),dtype=self.dtype_int)

        #Memory allocation for c-functions
        min_vals = np.empty(len(self.x),dtype=self.dtype)
        max_vals  = np.empty(len(self.x),dtype=self.dtype)
        min_mask = np.empty(len(self.x),dtype=self.dtype_int)
        max_mask = np.empty(len(self.x),dtype=self.dtype_int)
        search_box = np.empty((self.n_feat,2),dtype=self.dtype)
        search_pts_idx = np.empty(len(self.x),dtype=self.dtype_int)
        filter_pts_idx = np.empty(len(self.x),dtype=self.dtype_int)
        
        
        #Search filter which filters already found points for the next iterations
        search_filter = np.ones(len(self.y),dtype=bool)

        bboxes = []
        scanned_fidxs = []
        start = time.time()

        time_search = 0
        counter = 0
        
        '''
            Conditions 1) as long as one positive/rare instance is remaining 2) less than max boxes 3) as long as more than 1 sample is existing (for the rare case
            where only one 1 positive is remaining)
        '''
        while (np.sum(self.y[search_filter]) > 0) & (len(bboxes) < max_nbox) & (np.sum(search_filter) > 1):
            #filtered by already removed points
            labels = self.y[search_filter]
            data = self.x[search_filter]

            pts_idx.resize(len(labels),refcheck=False)
            min_vals.resize(len(labels),refcheck=False)
            max_vals.resize(len(labels),refcheck=False)
            min_mask.resize(len(labels),refcheck=False)
            max_mask.resize(len(labels),refcheck=False)
            search_pts_idx.resize(len(labels),refcheck=False)
            filter_pts_idx.resize(len(labels),refcheck=False)

            #random fraction of original feat_idxs
            rand_idx = np.random.choice(len(self.feat_idxs), max_evals, replace=False) 
            feat_idxs = self.feat_idxs[rand_idx]
            scanned_fidxs.extend(rand_idx)

            best_bbox = None
            best_score = np.inf
            best_feat_idxs = None

            pos_samples = np.where(labels == 1)[0].astype(self.dtype_int)
            pt_idx = pos_samples[np.random.choice(len(pos_samples),1)]

            for idxs in feat_idxs:
                seed = np.random.randint(100000000)

                feats = np.arange(self.n_feat,dtype=self.dtype_int)
                if rand_order == 1:
                    np.random.shuffle(feats)

                start_search = time.time()
                score = search_boxes(pt_idx,data[:,idxs],labels,feats,min_pts,early_stopping,class_weights,
                                    splitter,min_nobj,stop_infinite,stop_infinite_split,eps,seed,box,pts_idx,
                                    min_vals, max_vals,min_mask,max_mask,search_box, search_pts_idx,filter_pts_idx) # box and pts_idx are overwritten
                end_search = time.time()
                time_search += end_search - start_search

                if (score < best_score):
                    best_score = score
                    best_bbox = box.copy()
                    incl_pts = pts_idx.copy().astype(bool)
                    best_feat_idxs = idxs.copy()

            if best_score == np.inf:
                counter += 1
                if counter == retry_limit:
                    if self.verbose:
                        print("INFO: Stopped box search due to remaining heterogeneous duplicates!")
                    break 
            else:
                counter = 0

                mask = np.where(search_filter==True)[0][incl_pts]                

                if top_down:
                    b,mask = self.train_topdown(best_bbox,best_feat_idxs,mask,seed)
                    bboxes.extend(b)
                else:
                    bboxes.append([best_bbox,best_feat_idxs])

                #only rare instances removed from map or all 
                if del_nonrare == False:
                    rare_idx = np.where(self.y == 1)[0]
                    mask = np.intersect1d(mask,rare_idx)

                search_filter[mask] = False

                if self.verbose:
                    if top_down:
                        print("Included points: ",len(mask))
                    else:
                        print("Included points: ",np.sum(incl_pts))
                    print(best_bbox)
                    print(best_score)
                    print(best_feat_idxs)
                    print(f"Number of positive points left: {np.sum(self.y[search_filter])}")
                    print(f"####################################\n")

        end = time.time()
        if self.verbose:
            print(f"Time needed: {end-start}")
            print(f"Time needed for box search: {time_search}")
            print("Number of available features: ",len(np.unique(self.feat_idxs)))
            print("Number of scanned features: ",len(np.unique(self.feat_idxs[scanned_fidxs])))
            print("Number of boxes: ",len(bboxes))
        self.bboxes = bboxes

    '''
    Function to start the bottom-up construction phase of the DecisionBranch model 
    '''
    def train_boxSearch_parallel(self, n_jobs=5, max_evals="auto",max_nbox=100,min_nobj=1, splitter="half",min_pts=5,
                    early_stopping=True,class_weights=[1,1],del_nonrare=True,rand_order=True,stop_infinite_split="random",
                    stop_infinite=False,top_down=False,eps=1e-16,retry_limit=3):
        '''
            n_jobs: number of parallel jobs
            max_evals: number of evaluations among all available feature indices 
            max_nbox: maximum number of boxes 
            min_nobj:  minimum number of points in box
            splitter: splitting method to split (choose between half, min, max, random)
            min_pts: number of points to compare in each direction for expanding the boundary
            early_stopping : en-/disable early stopping (when no rare point exists in one direction during expansion)
            class_weight: class weights for both classes 
            del_nonrare: en-/disable if non-rare points should be removed when contained in box
            rand_order: en-/disable random order processing of the features within feature index
            stop_infinite: en-/disable expansion until infinity
            top_down: en-/disable top-down construction (to split impure box into multiple pure boxes)
            eps: epsilon factor to prevent numerical instability 
            retry_limit: number of retries when no box has been found that has better impurity than -infinity
        '''
        np.random.seed(self.seed)

        self.max_evals_input = max_evals

        #Random feature selection equal to Random Forest
        if max_evals == "auto":
            max_evals = int(np.sqrt(len(self.feat_idxs)))
        elif max_evals == "log2":
            max_evals = int(np.log2(len(self.feat_idxs)))
        elif type(max_evals) == float:
            max_evals = int(len(self.feat_idxs) * max_evals)
        elif max_evals == "all":
            max_evals = len(self.feat_idxs)
        if max_evals > len(self.feat_idxs):
            max_evals = len(self.feat_idxs)

        if min_pts > len(self.x):
            min_pts = len(self.x)

        if n_jobs > os.cpu_count():
            print("Info: n_jobs is larger than available CPUs! Set to n_cpus.")
            n_jobs = os.cpu_count()

        #in case only positive class weight is given
        if isinstance(class_weights,int):
            class_weights = [1,class_weights]
        
        self.cfg = {"n_jobs": n_jobs, "max_nbox": max_nbox, "min_nobj": min_nobj,
                    "max_evals": max_evals,"min_pts":min_pts,"early_stopping":early_stopping,
                    "splitter":splitter,"class_weights": class_weights,"del_nonrare":del_nonrare,
                   "rand_order": rand_order,"stop_infinite":stop_infinite,"stop_infinite_split":stop_infinite_split,"top_down":top_down,
                   "dtype":self.dtype,"eps":eps,"seed": self.seed}

        #Init for Cython functions
        early_stopping = int(early_stopping)
        stop_infinite = int(stop_infinite)
        rand_order = int(rand_order)
        stop_infinite_split = split_types[stop_infinite_split]
        splitter = split_types[splitter]
        class_weights = np.asarray(class_weights,dtype=self.dtype_int)

        #Parallelization
        if max_evals < n_jobs:
            n_jobs = max_evals

        #Search filter which filters already found points for the next iterations
        search_filter = np.ones(len(self.y),dtype=bool)

        bboxes = []
        scanned_fidxs = []
        start = time.time()

        time_search = 0
        counter = 0
        
        pool = mp.Pool(n_jobs,initializer=self._initializer_,initargs=(self.x,self.y,self.n_feat,self.dtype,self.dtype_int,))

        '''
            Conditions 1) as long as one positive/rare instance is remaining 2) less than max boxes 3) as long as more than 1 sample is existing (for the rare case
            where only one 1 positive is remaining)
        '''
        while (np.sum(self.y[search_filter]) > 0) & (len(bboxes) < max_nbox) & (np.sum(search_filter) > 1):
            #filtered by already removed points
            labels = self.y[search_filter]
            #data = self.x[search_filter]

            #random fraction of original feat_idxs
            rand_idx = np.random.choice(len(self.feat_idxs), max_evals, replace=False) 
            feat_idxs = self.feat_idxs[rand_idx]
            scanned_fidxs.extend(rand_idx)

            pos_samples = np.where(labels == 1)[0].astype(self.dtype_int)
            pt_idx = pos_samples[np.random.choice(len(pos_samples),1)]

            params = []
            for idxs in feat_idxs:
                seed = np.random.randint(100000000)
                feats = np.arange(self.n_feat,dtype=self.dtype_int)
                if rand_order == 1:
                    np.random.shuffle(feats)
                args = [idxs,search_filter,pt_idx,feats,min_pts,early_stopping,class_weights,splitter,
                        min_nobj,stop_infinite,stop_infinite_split,eps,seed]
                params.append(args)

            start_search = time.time()
            try:
                results = pool.map(self._search_box_,params)
            except Exception as e:
                print(f"Warning: Error in Box creation! \n {e}")
                pool.close()
                pool.join()
                raise ValueError("Error during parallel training execution!")
            end_search = time.time()
            time_search += end_search - start_search

            best_score = np.inf
            results_idx = None

            for i in range(len(results)):
                if results[i][0] < best_score:
                    best_score = results[i][0]
                    results_idx = i

            if results_idx is None:
                counter += 1
                if counter == retry_limit:
                    if self.verbose:
                        print("INFO: Stopped box search due to remaining heterogeneous duplicates!")
                    break
            else:
                counter = 0
                best_bbox = results[results_idx][1]
                incl_pts = results[results_idx][2]
                best_feat_idxs = results[results_idx][3]

                mask = np.where(search_filter==True)[0][incl_pts]
                
                if top_down:
                    b,mask = self.train_topdown(best_bbox,best_feat_idxs,mask,seed)
                    bboxes.extend(b)
                else:
                    bboxes.append([best_bbox,best_feat_idxs])

                #only rare instances removed from map or all 
                if del_nonrare == False:
                    rare_idx = np.where(self.y == 1)[0]
                    mask = np.intersect1d(mask,rare_idx)

                search_filter[mask] = False

                if self.verbose:
                    print("Included points: ",len(mask))
                    print(best_bbox)
                    print(best_score)
                    print(best_feat_idxs)
                    print(f"Number of positive points left: {np.sum(self.y[search_filter])}")
                    print(f"####################################\n")
        pool.close()
        pool.join()
        
        end = time.time()
        if self.verbose:
            print(f"Time needed: {end-start}")
            print(f"Time needed for box search: {time_search}")
            print("Number of available features: ",len(np.unique(self.feat_idxs)))
            print("Number of scanned features: ",len(np.unique(self.feat_idxs[scanned_fidxs])))
            print("Number of boxes: ",len(bboxes))
        self.bboxes = bboxes
    
    '''
    Function for starting the top-down construction phase (is called automatically
    when the parameter top_down = True in the bottom-construction)
    '''
    def train_topdown(self,box,feat_idxs,pts_idx,seed):
        y = self.y[pts_idx]

        if np.all(y == 1):
            return [[box,feat_idxs]],pts_idx
        else:
            x = self.x[pts_idx][:,feat_idxs]
            dtree = DecisionTreeClassifier(random_state=seed)

            dtree.fit(x,y)

            paths = []
            path = []
            
            tree = dtree.tree_

            def recurse(node, path, paths):
                if tree.children_left[node] != tree.children_right[node]:
                    p1, p2 = list(path), list(path)
                    p1.append([tree.feature[node],1,tree.threshold[node]]) # Feat idx, min/max, split value
                    recurse(tree.children_left[node], p1, paths)
                    p2.append([tree.feature[node],0,tree.threshold[node]])
                    recurse(tree.children_right[node], p2, paths)
                else:
                    if (tree.value[node][0][1] > 0) and (tree.value[node][0][0] == 0): #rare leaf
                        paths.append(path)

            recurse(0, path, paths)

            bboxes = []

            for leaf in paths:
                pure_box = box.copy()
                for split in leaf:
                    f_idx,direction,split_value = split
                    pure_box[f_idx,direction] = split_value
                bboxes.append([pure_box,feat_idxs])
            
            #Update pts_idx
            pts_idx = np.delete(pts_idx,np.where(y==0)[0])

            return bboxes,pts_idx

    '''
    Function for starting the top-down construction phase for the decision tree
    trained on all features
    '''
    def train_postTree(self,**kwargs):
        trees = []
        start = time.time()
        x,y = self.x, self.y

        box_pts = np.ones(len(x),dtype=bool)

        #Random feature selection equal to Random Forest
        max_evals = self.max_evals_input
        if max_evals == "auto":
            max_evals = int(np.sqrt(x.shape[1]))
        elif max_evals == "log2":
            max_evals = int(np.log2(x.shape[1]))
        elif type(x.shape[1]) == float:
            max_evals = int(x.shape[1] * max_evals)
        elif max_evals == "all":
            max_evals = x.shape[1]
        if max_evals > x.shape[1]:
            max_evals = x.shape[1]

        for i,b in enumerate(self.bboxes):
            box,feat_idx = b
            box_pts = np.asarray(box_pts,dtype=np.intc)
            filter_py(box,x[:,feat_idx],box_pts)
            box_pts = np.asarray(box_pts,dtype=bool)
            if y[box_pts].all() == 1:
                pure = True
            else: 
                pure = False

            if not pure:
                dtree = DecisionTreeClassifier(random_state=self.seed+i,max_features=max_evals,
                                                **kwargs)#,class_weight="balanced")
                dtree.fit(x[box_pts],y[box_pts])
                trees.append(dtree)
            else:
                trees.append(None)
        end = time.time()
        if self.verbose:
            print("-----------------------------")
            print("Post-Dtree:\nTime needed: ",end-start)
            print("")

        self.trees = trees
        self.postTree = True

    '''
        Function for performing the search based on the constructed boxes for given data x (
            x needs to have the same dimensionality than the training data)
        Returns indices of all rare/positive elements
    '''
    def search(self,x,box_ids="all"):
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

        if box_ids == "all":
            box_ids = np.arange(len(self.bboxes))
        elif type(box_ids) == int:
            box_ids = [box_ids]

        pts = []
        mask = np.ones(len(x),dtype=bool)
        for i in box_ids:
            box,feat_idxs = self.bboxes[i]
            output = np.ones(np.sum(mask),dtype=np.intc)
            filter_py(box,x[mask][:,feat_idxs],output)
            output = np.asarray(output,dtype=bool)
            if np.sum(output) > 0:
                is_empty = False
                if self.postTree:
                    tree = self.trees[i]
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

    '''
    Predict function similar to sklearn predict function that returns label 1/0 whether the point
    is inside or outside one of the boxes
    '''
    #Returns the predicted labels for each inserted sample
    def predict(self,x,box_ids="all"):
        if isinstance(x,pd.DataFrame):
            x = np.array(x,dtype=self.dtype)
        elif isinstance(x,torch.Tensor):
            x = x.numpy().astype(self.dtype)
        elif isinstance(x,np.ndarray):
            if x.dtype != self.dtype:
                x = x.astype(self.dtype)
        else:
            raise ValueError("No known data type for x!")
        pos_idx = self.search(x,box_ids)
        mask = np.zeros(len(x),dtype=int)
        if len(pos_idx) > 0:
            mask[pos_idx] = 1  
        return mask
    
    '''
    Function to output the found boxes during training
    '''
    def get_boxes(self):
        n_boxes = len(self.bboxes)
        mins = np.empty((n_boxes,self.n_feat),dtype=self.dtype)
        maxs = np.empty((n_boxes,self.n_feat),dtype=self.dtype)
        fidxs = np.empty((n_boxes,self.n_feat),dtype=self.dtype_int)
        for n,b in enumerate(self.bboxes):
            box,f_idx = b
            mins[n] = box[:,0]
            maxs[n] = box[:,1]
            fidxs[n] = f_idx
        fidxs = fidxs.tolist()
        return mins,maxs,fidxs

    @staticmethod
    def _initializer_(x,y,n_feat,dtype,dtype_int):
        global X,Y,min_vals,max_vals,min_mask,max_mask,search_box,search_pts_idx,filter_pts_idx,box,pts_idx
        X = x
        Y = y
        size = len(X)
        min_vals = np.empty(size,dtype=dtype)
        max_vals  = np.empty(size,dtype=dtype)
        min_mask = np.empty(size,dtype=dtype_int)
        max_mask = np.empty(size,dtype=dtype_int)
        search_box = np.empty((n_feat,2),dtype=dtype)
        search_pts_idx = np.empty(size,dtype=dtype_int)
        filter_pts_idx = np.empty(size,dtype=dtype_int)

        box = np.empty((n_feat,2),dtype=dtype)
        pts_idx = np.empty(size,dtype=dtype_int)


    @staticmethod
    def _search_box_(args):
        idxs,search_filter,pt_idx,feats,min_pts,early_stopping,class_weights,splitter,min_nobj,stop_infinite,stop_infinite_split,eps,seed = args
        np.random.seed(seed)
        data = X[search_filter][:,idxs]
        labels = Y[search_filter]

        score = search_boxes(pt_idx,data,labels,feats,min_pts,early_stopping,class_weights,
                                    splitter,min_nobj,stop_infinite,stop_infinite_split,eps,seed,box,pts_idx,
                                    min_vals, max_vals,min_mask,max_mask,search_box, search_pts_idx,filter_pts_idx) # box and pts_idx are overwritten
        pts_idx_copy = pts_idx[:len(labels)].astype(bool) # only until end of data is reached
        return score,box.copy(),pts_idx_copy,idxs
        




    



