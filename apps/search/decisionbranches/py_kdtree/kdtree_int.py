import time
import numpy as np
import math
import os
import pickle

from .cython.uint8.box_query import recursive_search_box_int64 as search_box_8_int64
from .cython.uint8.point_query import recursive_search_point_int64 as search_point_8_int64
from .cython.uint8.box_query import recursive_search_box_uint32 as search_box_8_uint32
from .cython.uint8.point_query import recursive_search_point_uint32 as search_point_8_uint32

MAX_INT = {"uint32":np.iinfo(np.uint32).max,"uint64":np.iinfo(np.uint64).max,
            "int32":np.iinfo(np.int32).max,"int64":np.iinfo(np.int64).max}

class KDTree():
    def __init__(self, path=None,dtype="uint8",dtype_int="uint32",leaf_size=30,model_file=None,inmemory=False,mmap_file=None,leaf_index_map_file=None,verbose=True):
        if path is None:
            path = os.getcwd()
        
        if not os.path.isdir(path):
            os.makedirs(path)

        self.path = path
        self.verbose = verbose
        self.dtype = dtype
        self.leaf_size = leaf_size
        self.dtype_int = dtype_int
        self.tree = None
        self.inmemory = inmemory

        if mmap_file is None:
            self.mmap_file = os.path.join(self.path,"map.mmap")
        else:
            self.mmap_file = os.path.join(self.path,mmap_file)
        
        if model_file is None:
            self.model_file = os.path.join(path,"tree.pkl")
        else:
            self.model_file = os.path.join(path,model_file)

        if leaf_index_map_file is None:
            self.leaf_index_map_file = os.path.join(self.path,"leaf_index_map.mmap")
        else:
            self.leaf_index_map_file = os.path.join(self.path,leaf_index_map_file)

        if os.path.isfile(self.model_file) and os.path.isfile(self.mmap_file) and os.path.isfile(self.leaf_index_map_file):
            if self.verbose:
                print(f"INFO: Load existing model under {self.model_file}")
            self._load()

            #keep track if the input leaf size matches the loaded 
            self.org_leaf_size = leaf_size
            if self.inmemory:
                self._data = np.memmap(self.mmap_file, dtype=self.dtype, mode='r', shape=self.mmap_shape)[:].copy()
                self._data_leaf_index = np.memmap(self.leaf_index_map_file, dtype=self.dtype_int, mode='r', shape=(self.n_leaves,self.leaf_size))[:].copy()

    '''
    Function fit() creates the k-d tree
    '''
    def fit(self, X,mmap_idxs=None,build_inmemory=False):
        '''
        mmap_idxs: whether X is stored in memory or in a mmap (in case of mmap -> list of dims as input)
        '''        
        if mmap_idxs is None:
            self._dim = len(X[0])
        else:
            self._dim = len(mmap_idxs)
        
        if (np.dtype(self.dtype) != X.dtype):
            print(f"WARNING: X dtype {X.dtype} does not match with Model dtype {self.dtype}")

        if self.tree is not None:
            if self.verbose:
                print("INFO: Model is already loaded, overwrite existing model!")
            os.remove(self.model_file)
            os.remove(self.mmap_file)
            os.remove(self.leaf_index_map_file)

            self.leaf_size = self.org_leaf_size

        if len(X) >= MAX_INT[self.dtype_int]:
            raise Exception("Number of points exceeds maximum of dtype_int!")
            #self.dtype_int = "uint64"
        # if len(X) < 2_147_483_647:
        #     self.dtype_int = "int32"
        # else:
        #     self.dtype_int = "int64"
        I = np.arange(len(X),dtype=self.dtype_int)

        
        self.depth = self._calc_depth(len(X))
        #update the leaf size with the actual value
        self.leaf_size = int(np.ceil(len(X) / 2**self.depth))
        self.n_leaves = 2**self.depth
        self.n_nodes = 2**(self.depth+1)-1
        self.tree = np.empty((self.n_nodes,self._dim,2),dtype=self.dtype)
        self.mmap_shape = (self.n_leaves,self.leaf_size,self._dim)

        if build_inmemory:
            mmap = np.empty(self.mmap_shape,dtype=self.dtype)*-1
            leaf_index_map = np.empty((self.n_leaves,self.leaf_size),dtype=self.dtype_int)
        else:
            mmap = np.memmap(self.mmap_file, dtype=self.dtype, mode='w+', shape=self.mmap_shape)
            leaf_index_map = np.memmap(self.leaf_index_map_file, dtype=self.dtype_int, mode='w+', shape=(self.n_leaves,self.leaf_size))
        #np.ones((self.n_leaves,self.leaf_size),dtype=self.dtype_int)*-1

        start = time.time()
        if mmap_idxs is None:
            self._build_tree(X, I,mmap,leaf_index_map)
        else:
            self._build_tree_mmap(X,I,mmap,leaf_index_map,mmap_idxs)
        end = time.time()
        self._save()

        if self.inmemory:
            self._data = mmap[:].copy()
            self._data_leaf_index = leaf_index_map[:].copy()

        if build_inmemory:
            mmap_file = np.memmap(self.mmap_file, dtype=self.dtype, mode='w+', shape=self.mmap_shape)
            leaf_index_map_file = np.memmap(self.leaf_index_map_file, dtype=self.dtype_int, mode='w+', shape=(self.n_leaves,self.leaf_size))

            mmap_file[:] = mmap[:]
            leaf_index_map_file[:] = leaf_index_map[:]
            mmap_file._mmap.close()
            leaf_index_map_file._mmap.close()
        else:
            mmap._mmap.close()
            leaf_index_map._mmap.close()

        if self.inmemory:
            self._data = mmap[:].copy()
            self._data_leaf_index = leaf_index_map[:].copy()

        if self.verbose:
            print(f"INFO: Building tree took {end-start} seconds")

    
    '''
    Cythonic query_box function that is more efficient than the Python variant but so far
    only returns the contained indices of the points
    '''
    def query_box_cy(self,mins,maxs,max_pts=0,max_leaves=0,mem_cap=0.001):
        '''
        mins: 1D-array of lower boundaries of the hyperrectangle
        maxs: 1D-array of upper boundaries of the hyperrectangle
        max_pts: stopps after number of points
        max_leaves: stopps after number of loaded leaves
        mem_cap: proportion of the total number of points that is expected to be returned by the query 
        '''
        if self.tree is None:  
            raise Exception("Tree not fitted yet!")
        
        if (max_leaves > 0) and (max_pts > 0):
            raise Exception("Only one max parameter allowed at once!")

        if self.inmemory:
            mmap = self._data
            leaf_index_map = self._data_leaf_index
        else:
            mmap = np.memmap(self.mmap_file, dtype=self.dtype, mode='r', shape=self.mmap_shape)
            leaf_index_map = np.memmap(self.leaf_index_map_file, dtype=self.dtype_int, mode='r', shape=(self.n_leaves,self.leaf_size))

        if isinstance(mins,list):
            mins = np.array(mins,dtype=self.dtype)
        if isinstance(maxs,list):
            maxs = np.array(maxs,dtype=self.dtype)


        arr_loaded = np.empty(1,dtype=np.intc)
        start = time.time()


        if max_pts > 0 or max_leaves > 0:
            print("WARNING: max_pts and max_leaves are not supported for uint8 dtypes!")
        
        if self.dtype_int == "uint32":
            indices = search_box_8_uint32(mins,maxs,self.tree,leaf_index_map,self.n_leaves,self.n_nodes,mmap,max_pts,max_leaves,mem_cap,arr_loaded)
        elif self.dtype_int == "int64":
            indices = search_box_8_int64(mins,maxs,self.tree,leaf_index_map,self.n_leaves,self.n_nodes,mmap,max_pts,max_leaves,mem_cap,arr_loaded)
        else:
            raise Exception(f"dtype_int {self.dtype_int} not supported!")
    
        end = time.time()
        if self.verbose:
            print(f"INFO: Query took: {end-start} seconds")
            print(f"INFO: Query loaded leaves: {arr_loaded[0]}")

        mmap._mmap.close()
        leaf_index_map._mmap.close()
        return indices.base,end-start,arr_loaded[0]

    def query_point_cy(self,point,k=5,stop_leaves=None):
        if self.tree is None:  
            raise Exception("Tree not fitted yet!")

        if self.inmemory:
            mmap = self._data
            leaf_index_map = self._data_leaf_index
        else:
            mmap = np.memmap(self.mmap_file, dtype=self.dtype, mode='r', shape=self.mmap_shape)
            leaf_index_map = np.memmap(self.leaf_index_map_file, dtype=self.dtype_int, mode='r', shape=(self.n_leaves,self.leaf_size))

        if stop_leaves is None:
            stop_leaves = 2**self.depth
        else:
            if k > self.leaf_size * stop_leaves:
                raise Exception("k is larger than the number of points visited! Adjust k or stop_leaves!")

        start = time.time()

        indices = np.empty(k,dtype=self.dtype_int)
        distances = np.empty(k,dtype="float64")

        if self.dtype_int == "uint32":
            leaves_visited = search_point_8_uint32(point,k,self.tree,leaf_index_map,self.n_leaves,self.n_nodes,stop_leaves,mmap,indices,distances)
        elif self.dtype_int == "int64":
            leaves_visited = search_point_8_int64(point,k,self.tree,leaf_index_map,self.n_leaves,self.n_nodes,stop_leaves,mmap,indices,distances)
        else:
            raise Exception(f"dtype_int {self.dtype_int} not supported!")

        order = np.argsort(distances)
        indices = indices[order]
        distances = distances[order]
        end = time.time()
        if self.verbose:
            print(f"INFO: Query search took: {end-start} seconds")
            print(f"INFO: Query visited {leaves_visited} leaves")

        return indices,distances,end-start,leaves_visited

    def _build_tree(self, pts, indices,mmap,leaf_index_map, depth=0,idx=0):
        #if root
        if idx == 0: 
            self.tree[idx] = np.array([[0,255]]*self._dim)

        bounds = self.tree[idx]

        if len(pts) <= self.leaf_size: 
            if pts.dtype != np.dtype(self.dtype):
                pts = pts.astype(self.dtype)
            
            lf_idx = self.n_leaves+idx-self.n_nodes

            #shape = pts.shape
            #if shape[0] != self.leaf_size:
                # nan = np.array([*[0]*self._dim],dtype=self.dtype)
                # pts = np.vstack([pts,nan])

            leaf_index_map[lf_idx,:len(indices)] = indices
            leaf_index_map[lf_idx,len(indices):] = MAX_INT[self.dtype_int]

            mmap[lf_idx,:len(pts)] = pts[:]
            return 
        
        axis = depth % self._dim
        
        pts_ax = pts[:,axis]
        # if pts_ax.dtype != np.dtype(self.dtype):
        #     pts_ax = pts_ax.astype(self.dtype)
        part = pts_ax.argsort()
        indices = indices[part]
        pts = pts[part]
        pts_ax = pts_ax[part]

        midx = math.floor(len(pts)/2)
        median = pts_ax[midx]

        l_bounds,r_bounds = bounds.copy(),bounds.copy()
        l_bounds[axis,1] = median
        r_bounds[axis,0] = median

        l_idx,r_idx = self._get_child_idx(idx)

        self.tree[l_idx] = l_bounds
        self.tree[r_idx] = r_bounds

        self._build_tree(pts[:midx,:], indices[:midx],mmap,leaf_index_map, depth+1,l_idx)
        self._build_tree(pts[midx:,:], indices[midx:],mmap,leaf_index_map, depth+1,r_idx)

    def _build_tree_mmap(self, pts, indices,mmap,leaf_index_map, pts_mmap_idxs,depth=0,idx=0):
            #if root
        if idx == 0: 
            self.tree[idx] = np.array([[0,255]]*self._dim,dtype=self.dtype)
        else:
            indices.sort() #important for contiguous read

        bounds = self.tree[idx]

        if len(indices) <= self.leaf_size:
            pts_sub = pts[indices,:][:,pts_mmap_idxs]
            pts = np.c_[indices,pts_sub]

            if pts.dtype != np.dtype(self.dtype):
                pts = pts.astype(self.dtype)

            
            
            lf_idx = self.n_leaves+idx-self.n_nodes

            #shape = pts.shape
            # if shape[0] != self.leaf_size:
            #     nan = np.array([0,*[0]*self._dim],dtype=self.dtype)
            #     pts = np.vstack([pts,nan])
            leaf_index_map[lf_idx,:len(indices)] = indices
            leaf_index_map[lf_idx,len(indices):] = MAX_INT[self.dtype_int]
                
            mmap[lf_idx,:len(pts)] = pts[:]
            return 
        
        axis = depth % self._dim
        pts_axis = pts_mmap_idxs[axis]
        

        #Load into memory
        if pts.flags["C_CONTIGUOUS"]:
            pts_ax = self._load_col_from_mmap(pts,indices,pts_axis)
        elif pts.flags["F_CONTIGUOUS"]:
            pts_ax = pts[:,pts_axis]
            pts_ax = pts_ax[indices]

        part = pts_ax.argsort()
        indices = indices[part]
        pts_ax = pts_ax[part]

        midx = math.floor(len(pts_ax)/2)
        median = pts_ax[midx]

        l_bounds,r_bounds = bounds.copy(),bounds.copy()
        l_bounds[axis,1] = median
        r_bounds[axis,0] = median

        l_idx,r_idx = self._get_child_idx(idx)

        self.tree[l_idx] = l_bounds
        self.tree[r_idx] = r_bounds

        del pts_ax
        del part

        self._build_tree_mmap(pts, indices[:midx],mmap,leaf_index_map,pts_mmap_idxs, depth+1,l_idx)
        self._build_tree_mmap(pts, indices[midx:],mmap,leaf_index_map,pts_mmap_idxs, depth+1,r_idx)

    def _load(self):
        with open(self.model_file, 'rb') as file:
            new = pickle.load(file)

        self.tree = new["tree"]

        #TODO merge attributes into one
        self.depth = new["depth"]
        self.leaf_size = new["leaf_size"]
        self.n_leaves = new["n_leaves"]
        self.n_nodes = new["n_nodes"]
        self._dim = new["_dim"]
        self.mmap_shape = new["mmap_shape"]
        self.dtype = str(self.tree.dtype)
        self.dtype_int = new["dtype_int"]

    def _save(self):
        save_dict = {}
        for attribute in dir(self):
            attr_value = getattr(self, attribute)
            if not callable(attr_value) and not attribute.startswith("__"):
                save_dict[attribute] = attr_value

        with open(self.model_file, 'wb') as file:
            pickle.dump(save_dict, file) 
        
        # np.save(self.leaf_index_map_file,self.leaf_index_map)
        if self.verbose:            
            print(f"Model was saved under {self.model_file}")

    def _calc_depth(self,n):
        d = 0
        while n/2**d > self.leaf_size:
            d += 1
        return d
    
        # '''
    # Pythonic query_box function that returns the contained indices of the points + their
    # features
    # '''
    # def query_box(self,mins,maxs,index_only=False):
    #     '''
    #     mins: 1D-array of lower boundaries of the hyperrectangle
    #     maxs: 1D-array of upper boundaries of the hyperrectangle
    #     index_only: if True the features are not returned
    #     '''
    #     if self.tree is None:  
    #         raise Exception("Tree not fitted yet!")

    #     self._leaves_visited = 0
    #     self._loading_time = 0.

    #     start = time.time()
    #     indices,points = self._recursive_search(0,mins,maxs,index_only=index_only)
    #     end = time.time()
    #     if self.verbose:
    #         print(f"INFO: Query took: {end-start} seconds")
    #         print(f"INFO: Query loaded {self._leaves_visited} leaves")
    #         print(f"INFO: Query took {self._loading_time} seconds loading leaves")

    #     if len(indices) > 0:
    #         inds = np.concatenate(indices).astype(np.int64)
    #     else:
    #         inds = np.empty((0,),dtype=np.int64)

    #     if index_only: 
    #         return (inds,self._leaves_visited,end-start,self._loading_time)
    #     else:
    #         if len(points) > 0:
    #             pts = np.concatenate(points).astype(self.dtype)
    #         else:
    #             pts = np.empty((0,self._dim))
    #         return (inds,pts,self._leaves_visited,end-start,self._loading_time)
        
    
    # def query_point(self,point,k=5):
    #     if self.tree is None:  
    #         raise Exception("Tree not fitted yet!")

    #     if self.inmemory:
    #         mmap = self._data
    #     else:
    #         mmap = np.memmap(self.mmap_file, dtype=self.dtype, mode='r', shape=self.mmap_shape)

    #     start = time.time()

    #     indices = np.empty(k,dtype=np.int64)
    #     distances = np.empty(k,dtype=self.dtype)

    #     distances[:] = 255
    #     indices,distances = self._query_point(0,0,point,k,indices,distances,mmap)
        
    #     order = np.argsort(distances)
    #     indices = indices[order]
    #     distances = distances[order]
    #     end = time.time()
    #     if self.verbose:
    #         print(f"INFO: Query search took: {end-start} seconds")

    #     return indices,distances,end-start

    # def _recursive_search(self,idx,mins,maxs,indices=None,points=None,index_only=False):
    #     if points is None:
    #         points = []
    #     if indices is None:
    #         indices = []

    #     l_idx,r_idx = self._get_child_idx(idx)
        
    #     if (l_idx >= len(self.tree)) and (r_idx >= len(self.tree)):
    #         lf_idx = self.n_leaves+idx-self.n_nodes
    #         start = time.time()
    #         pts = self._get_pts(lf_idx)
    #         end = time.time()
    #         self._loading_time += end-start
    #         #also includes points on the borders of the box!
    #         mask = (np.all(pts[:,1:] >= mins,axis=1) ) &  (np.all(pts[:,1:] <= maxs, axis=1))
    #         indices.append(pts[:,0][mask].astype(np.int64))
    #         if not index_only:
    #             points.append(pts[:,1:][mask])
    #         return indices,points


        # l_bounds = self.tree[l_idx]
        # r_bounds = self.tree[r_idx]

        # #if at least intersects
        # if (np.all(l_bounds[:,1] >= mins )) and (np.all(maxs >= l_bounds[:,0])):
        #     self._recursive_search(l_idx,mins,maxs,indices,points)

        # if (np.all(r_bounds[:,1] >= mins )) and (np.all(maxs >= r_bounds[:,0])): 
        #     self._recursive_search(r_idx,mins,maxs,indices,points)

        # return indices,points

    # def _query_point(self,node_idx,depth,point,k,indices,distances,mmap):
    #     l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    #     ############################## Leaf ##########################################################################
    #     if (l_idx >= self.tree.shape[0]) and (r_idx >= self.tree.shape[0]):
    #         #calculate distance for each contained point and check whether it is smaller than the ones found so far
    #         lf_idx = self.n_leaves+node_idx-self.n_nodes
    #         for j in range(mmap.shape[1]):
    #             if j == mmap.shape[1]-1:
    #                 if mmap[lf_idx,j,0] == -1.:
    #                     continue
    #             dist = np.linalg.norm(mmap[lf_idx,j,1:] - point)
    #             max_dist = np.max(distances)
    #             if dist < max_dist:
    #                 dist_idx = np.argmax(distances)
    #                 distances[dist_idx] = dist
    #                 indices[dist_idx] = int(mmap[lf_idx,j,0])
    #         return indices,distances
        
    #     else:
    #         axis = depth % self.tree.shape[1]

    #         median = self.tree[l_idx][axis][1] #self.tree[node_idx][axis][1]
    #         if point[axis] < median:
    #             first = l_idx
    #             second = r_idx
    #         else:
    #             first = r_idx
    #             second = l_idx
    #         indices,distances = self._query_point(first,depth+1,point,k,indices,distances,mmap)
            
    #         max_dist = np.max(distances)
    #         max_dist_sub = abs(median - point[axis])
    #         if max_dist_sub < max_dist:
    #             indices,distances = self._query_point(second,depth+1,point,k,indices,distances,mmap)

    #         return indices,distances


    # def _get_pts(self,lf_idx):
    #     self._leaves_visited += 1
    #     fp = np.memmap(self.mmap_file, dtype=self.dtype, mode='r', shape=self.mmap_shape)
    #     leaf = fp[lf_idx,:,:]
    #     if leaf[-1,0] == -1:
    #         return leaf[:-1,:]
    #     else: 
    #         return leaf

    #Returns config parameters of model required to load the model
    def get_file_cfg(self):
        return {"path":self.path,"mmap_file":self.mmap_file,"model_file":self.model_file,
                "verbose":self.verbose} 
            
    @staticmethod
    def _get_child_idx(i):
        return (2*i)+1, (2*i)+2
    
    @staticmethod
    def _load_col_from_mmap(mmap,idxs,axis,chunksize=2048,verbose=True):
        pts = np.zeros(len(idxs),dtype=mmap.dtype)
        idx = 0

        while idx < len(idxs):
            chunk = idxs[idx:idx+chunksize]
            pts[idx:idx+chunksize] = mmap[chunk,axis]
            idx += chunksize
        
            if verbose:
                if (idx % chunksize**2) == 0:
                    print(idx)

        return pts


        


            
        


