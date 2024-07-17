import numpy as np
import torch
import os
import time
import sys
import os 
#import multiprocessing
from multiprocessing.pool import ThreadPool
from .kdtree import KDTree


class KDTreeSet():
    def __init__(self,indexes,path=None,dtype="float64",model_name="tree.pkl",verbose=True,group_prefix="",**kwargs):       
        if isinstance(indexes,np.ndarray):
            if len(indexes.shape) == 2:
                indexes = indexes.tolist()
            else:
                raise Exception("Indexes needs to be a 2D array!")
        elif not isinstance(indexes,list):
            raise Exception("No known datatype for indexes")

        inds = []
        for i in indexes:
            if len(set(i)) == len(i):
                i_new = sorted(i)
                if i_new not in inds:
                    inds.append(i_new)
        if len(inds) > 0:
            self.indexes = inds
        else:
            raise Exception("No valid indexes given!")

        self.n = len(indexes)
        self.model_name = model_name
        self.verbose = verbose
        self.dtype = dtype
        self.group_prefix = group_prefix

        self.trees = {}

        self.path = path
        if path is None:
            self.path = os.getcwd()
                
        self.trees = {}

        for i in indexes:
            dname = "_".join([group_prefix + str(j) for j in i])
            full = os.path.join(path,dname)
            tree = KDTree(path=full,dtype=dtype,model_file=model_name,verbose=self.verbose,**kwargs)
            self.trees[dname] = tree  


    def __len__(self):
        return self.n

    '''
    Function for creating set of k-d tree indices where X can fit into memory.
    '''
    def fit(self,X,mmap=False):    
        for i in self.indexes:
            dname = "_".join([self.group_prefix + str(j) for j in i])
            if self.trees[dname].tree is not None:
                if self.verbose:
                    print("INFO: Skip tree fit, model already existing - Change <path> in case of a new model!")
            else:
                if self.verbose:
                    print(f"INFO: model {dname} is trained")
                if not mmap:
                    if type(X) == np.memmap:
                        X_sub = X[:,i]
                        self.trees[dname].fit(X_sub)
                    else:
                        self.trees[dname].fit(X[:,i])
                else:
                    self.trees[dname].fit(X,mmap_idxs=i)

    '''
    Function for creating set of k-d tree indices where X cannot fit into memory and therefore needs to be loaded
    sequentially from disk and stored in mmap.
    '''
    def fit_seq_mmap(self,X_parts_list,size,n_cached=1,parts_path=None,mmap_path=None,data_driver="npy"):
        '''
        X_parts_list: list of filenames that contain the data chunks
        size: total number of points
        data_driver: either data can be stored via npy (numpy) or torch
        '''
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"

        if parts_path is None:
            parts_path = self.path

        if mmap_path is None:
            mmap_path = os.path.join(self.path,"tmp_mmap.mmap")
        else:
            mmap_path = os.path.join(mmap_path,"tmp_mmap.mmap")

        #Filter trained idxs
        idxs = []
        for i in self.indexes:
            dname = "_".join([self.group_prefix + str(j) for j in i])
            if self.trees[dname].tree is None:
                idxs.append(i)     

        if len(idxs) > 0:
            c = 0
            while c < len(idxs):
                sub = idxs[c:c+n_cached]
                flat_idx = [item for sublist in sub for item in sublist] #Flatten list

                mmap = np.memmap(mmap_path, dtype=self.dtype, mode='w+', shape=(size,len(flat_idx)))
                pointer = 0
                for f in X_parts_list:
                    if self.verbose:
                        print("INFO: Load file ",f)
                    fname = os.path.join(parts_path,f)
                    if data_driver == "npy":
                        x = np.load(fname)[:,flat_idx]
                    elif data_driver == "torch":
                        x = torch.load(fname)[:,flat_idx].detach().numpy()
                    else:
                        raise Exception(f"Not existing data driver {data_driver}!")
                    if x.dtype != np.dtype(self.dtype):
                        x = x.astype(self.dtype)
                    mmap[pointer:pointer+len(x),:] = x
                    pointer += len(x)

                #Train models
                for i in range(len(sub)):
                    start = len([item for sublist in sub[:i] for item in sublist])
                    end = start+len(sub[i])
                    dname = dname = "_".join([self.group_prefix + str(j) for j in sub[i]])
                    if self.verbose:
                        print(f"INFO: Model {dname} is trained")
                    self.trees[dname].fit(mmap,mmap_idxs=list(range(start,end)))
                c += n_cached
                os.remove(mmap_path)
        else:
            if self.verbose:
                print("INFO: Skipping train as the model has already been trained! Change model_file in case of a new model!")

    '''
    Function for querying single tree in treeset via Python function
    '''
    def query(self,mins,maxs,idx):
        '''
        mins: 1D-array of lower boundaries of the hyperrectangle
        maxs: 1D-array of upper boundaries of the hyperrectangle
        idx: list containing the feature indices of the hyperrectangle
        '''
        if mins.dtype != np.dtype(self.dtype):
            mins = mins.astype(self.dtype)
        if maxs.dtype != np.dtype(self.dtype):
            maxs = maxs.astype(self.dtype)

        dname = "_".join([self.group_prefix + str(j) for j in sorted(idx)])
        inds, pts,leaves_visited,time,loading_time = self.trees[dname].query_box(mins,maxs)

        return (inds,pts,leaves_visited,time,loading_time)

    '''
    Function for querying single tree in treeset via Cython function
    '''
    def query_cy(self,mins,maxs,idx):
        '''
        mins: 1D-array of lower boundaries of the hyperrectangle
        maxs: 1D-array of upper boundaries of the hyperrectangle
        idx: list containing the feature indices of the hyperrectangle
        '''
        if mins.dtype != np.dtype(self.dtype):
            mins = mins.astype(self.dtype)
        if maxs.dtype != np.dtype(self.dtype):
            maxs = maxs.astype(self.dtype)

        dname = "_".join([self.group_prefix + str(j) for j in sorted(idx)])
        inds, time,loaded_leaves = self.trees[dname].query_box_cy(mins,maxs)

        return (inds,time,loaded_leaves)

    '''
    Function for performing multiple box queries simultanously returning the indices of the points + their features
    Function multi_query works slower than the multi_query_ranked since it also filters for the complete found points
    '''
    def multi_query(self,mins,maxs,idxs,no_pts=False,n_jobs=-1):
        '''
        mins and maxs : arrays or lists of min/max boundaries (in 2D array format) 
        idxs: list containing lists of feature indices of the hyperrectangles
        no_pts: does not return the features of the found points
        '''
        if isinstance(mins,np.ndarray):
            if mins.dtype != np.dtype(self.dtype):
                mins = mins.astype(self.dtype)

        if isinstance(maxs,np.ndarray):       
            if maxs.dtype != np.dtype(self.dtype):
                maxs = maxs.astype(self.dtype)

        start = time.time()

        if n_jobs == -1:
            total_cpus = os.cpu_count()
            if total_cpus > len(idxs):
                n_jobs = len(idxs)
            else:
                n_jobs = total_cpus
        else:
            n_jobs = n_jobs

        params = self._create_params(mins,maxs,idxs,False,False)

        #pool = multiprocessing.Pool(n_jobs)
        pool = ThreadPool(n_jobs)

        try:
            results = pool.starmap(_static_query,params)
        except  Exception as e:
            print(f"Warning: Error in query! \n {e}")
            pool.close()
            sys.exit()
        pool.close()
        pool.join()

        i_list = []
        # To account for returned pts of different dimensionality 
        p_list = []

        leaves_visited = 0
        loading_time = 0.
        for i in range(len(idxs)):
            inds, pts,lv,_,lt = results[i]
            #get inds not part of i_list so far
            new_idx = np.arange(len(inds))
            if len(i_list) > 0:
                new_idx = np.where(np.in1d(inds,i_list) == False)
            i_list.extend(inds[new_idx].tolist())
            if no_pts == False:
                p_list.append(pts[new_idx])
            leaves_visited += lv
            loading_time += lt
        end = time.time()
        if self.verbose:
            print("#############################################")
            print(f"INFO: Query finished in {end-start} seconds")
            print(f"INFO: Query loaded {leaves_visited} leaves")
            print(f"INFO: Query loading time: {loading_time} s")

        points = np.concatenate(p_list)
        inds = np.array(i_list,dtype=np.int64)
        return (inds,points,leaves_visited,end-start,loading_time)

    '''
    Function for performing multiple box queries simultanously returning only the indices of the found points +
    the counts of how many times they were found within all boxes - Python version
    '''
    def multi_query_ranked(self,mins,maxs,idxs):
        '''
        mins and maxs : arrays or lists of min/max boundaries (in 2D array format) 
        idxs: list containing lists of feature indices of the hyperrectangles
        no_pts: does not return the features of the found points
        '''
        if isinstance(mins,np.ndarray):
            if mins.dtype != np.dtype(self.dtype):
                mins = mins.astype(self.dtype)

        if isinstance(maxs,np.ndarray):       
            if maxs.dtype != np.dtype(self.dtype):
                maxs = maxs.astype(self.dtype)

        start = time.time()

        inds = []

        leaves_visited = 0
        loading_time  = 0.
        for i in range(len(idxs)):
            dname = "_".join([self.group_prefix + str(j) for j in sorted(idxs[i])])
            i, lv,_,lt = self.trees[dname].query_box(mins[i],maxs[i],index_only=True)
            inds.append(i)
            leaves_visited += lv
            loading_time += lt

        inds = np.concatenate(inds)
        inds, counts = np.unique(inds,return_counts=True)

        order = np.argsort(-counts)
                
        end = time.time()
        if self.verbose:
            print("#############################################")
            print(f"INFO: query finished in {end-start} seconds")
            print(f"INFO: Query loaded {leaves_visited} leaves")
            print(f"INFO: Query loading time: {loading_time} s")

        return (inds[order],counts[order],leaves_visited,end-start,loading_time)

    '''
    Function for performing multiple box queries simultanously returning only the indices of the found points +
    the counts of how many times they were found within all boxes - Cython version
    '''
    def multi_query_ranked_cy(self,mins,maxs,idxs):
        if isinstance(mins,np.ndarray):
            if mins.dtype != np.dtype(self.dtype):
                mins = mins.astype(self.dtype)

        if isinstance(maxs,np.ndarray):       
            if maxs.dtype != np.dtype(self.dtype):
                maxs = maxs.astype(self.dtype)

        start = time.time()

        inds = []

        loaded_leaves = 0
        for i in range(len(idxs)):
            dname = "_".join([self.group_prefix + str(j) for j in sorted(idxs[i])])
            i,_,leaves = self.trees[dname].query_box_cy(mins[i],maxs[i])
            loaded_leaves += leaves
            inds.append(i)

        inds = np.concatenate(inds)
        inds, counts = np.unique(inds,return_counts=True)
        order = np.argsort(-counts)
                
        end = time.time()
        if self.verbose:
            print("#############################################")
            print(f"INFO: Query finished in {end-start} seconds")
            print(f"INFO: Query loaded {loaded_leaves} leaves")


        return (inds[order],counts[order],end-start,loaded_leaves)

    '''
    Function that returns all trees of the treeset that are built.
    '''
    def get_fitted_trees(self,array=False):
        fitted_trees = []
        for k,v in self.trees.items():
            if v.tree is not None:
                fitted_trees.append(k)
        if array:
            arr = np.array([np.array(i.split("_"),dtype=int) for i in fitted_trees])
            return arr
        return fitted_trees

    def _create_params(self,mins,maxs,idxs,index_only,cython=False):
        params = []
        for i in range(len(idxs)):
            dname = "_".join([self.group_prefix + str(j) for j in sorted(idxs[i])])
            cfg_dict = self.trees[dname].get_file_cfg()
            if cython:
                params.append([cfg_dict,mins[i],maxs[i],index_only,cython])
            else:
                params.append([cfg_dict,mins[i],maxs[i],index_only])
        return params

def _static_query(cfg,mins,maxs,index_only,cython):
    #To be executed silently
    cfg["verbose"] = False
    tree = KDTree(**cfg)
    
    if cython:
        return tree.query_box_cy(mins,maxs)
    else:
        return tree.query_box(mins,maxs,index_only)
    






