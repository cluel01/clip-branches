# cython: profile=False
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdlib cimport malloc, free, realloc

        
cimport cython
#from cython.parallel import prange
import numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef long[::1] recursive_search_box(float[::1] mins,float[::1] maxs, float[:,:,::1] tree,int n_leaves,
                    int n_nodes,const float[:,:,::1] mmap,int max_pts,int max_leaves,double mem_cap,int[::1] arr_loaded):    
    cdef long[::1] indices_view
    cdef long ind_len = int(mmap.shape[0]*mmap.shape[1]*mem_cap) 
    cdef long extend_mem = ind_len

    cdef long ind_pt = 0 
    cdef long* indices = <long*> malloc(ind_len * sizeof(long))

    cdef int loaded_leaves = 0

    try:
        if max_pts > 0:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit(0,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,0)
        elif max_leaves > 0:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit_leaves(0,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_leaves,loaded_leaves,0)
        else:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box(0,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)
        arr_loaded[0] = loaded_leaves
        indices_view = np.empty(ind_pt,dtype=np.int64)
        for i in range(ind_pt):
            indices_view[i] = indices[i]
        return indices_view 
    finally:
        free(indices)   

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long*,long,long,int) _recursive_search_box(int node_idx,float[::1] mins,float[::1] maxs, float[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, long ind_pt,long ind_len,const float[:,:,::1] mmap,long extend_mem, int loaded_leaves, int contained) nogil:
    cdef int l_idx, r_idx,intersects, ret,lf_idx,isin,j,k
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef float[:,:] bounds,l_bounds,r_bounds
    cdef float leaf_val
    
    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves+node_idx-n_nodes
        loaded_leaves += 1
        if contained == 1:
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if mmap[lf_idx,j,0] == -1.:
                        continue
                indices[ind_pt] = int(mmap[lf_idx,j,0])
                ind_pt += 1

                if ind_pt == ind_len:
                    indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                    ind_len += extend_mem
        else:
            for j in range(mmap.shape[1]):
                k = 0
                isin = 0
                while (k < mmap.shape[2]-1) and (isin == k):
                    if j == mmap.shape[1]-1:
                        if mmap[lf_idx,j,0] == -1.:
                            k += 1
                            continue
                    leaf_val = mmap[lf_idx,j,k+1]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = int(mmap[lf_idx,j,0])
                    ind_pt += 1
                    if ind_pt == ind_len:
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)
            
    return indices,ind_pt,ind_len,loaded_leaves

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long*,long,long,int) _recursive_search_box_limit(int node_idx,float[::1] mins,float[::1] maxs, float[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, long ind_pt,long ind_len,const float[:,:,::1] mmap,long extend_mem, int max_pts,int loaded_leaves, int contained) nogil:
    cdef int l_idx, r_idx,intersects, ret,lf_idx,isin,j,k
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef float[:,:] bounds,l_bounds,r_bounds
    cdef float leaf_val
    
    if ind_pt == max_pts:
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves+node_idx-n_nodes
        loaded_leaves += 1
        if contained == 1:
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if mmap[lf_idx,j,0] == -1.:
                        continue
                indices[ind_pt] = int(mmap[lf_idx,j,0])
                ind_pt += 1
                if ind_pt == max_pts:
                    return indices,ind_pt,ind_len,loaded_leaves
                if ind_pt == ind_len:
                    indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                    ind_len += extend_mem
        else:
            for j in range(mmap.shape[1]):
                k = 0
                isin = 0
                while (k < mmap.shape[2]-1) and (isin == k):
                    if j == mmap.shape[1]-1:
                        if mmap[lf_idx,j,0] == -1.:
                            k += 1
                            continue
                    leaf_val = mmap[lf_idx,j,k+1]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = int(mmap[lf_idx,j,0])
                    ind_pt += 1

                    if ind_pt == max_pts:
                        return indices,ind_pt,ind_len,loaded_leaves

                    if ind_pt == ind_len:
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,0)
            
    return indices,ind_pt,ind_len,loaded_leaves

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long*,long,long,int) _recursive_search_box_limit_leaves(int node_idx,float[::1] mins,float[::1] maxs, float[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, long ind_pt,long ind_len,const float[:,:,::1] mmap,long extend_mem, int max_leaves,int loaded_leaves, int contained) nogil:
    cdef int l_idx, r_idx,intersects, ret,lf_idx,isin,j,k
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef float[:,:] bounds,l_bounds,r_bounds
    cdef float leaf_val
    
    if loaded_leaves == max_leaves:
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves+node_idx-n_nodes
        loaded_leaves += 1
        if contained == 1:
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if mmap[lf_idx,j,0] == -1.:
                        continue
                indices[ind_pt] = int(mmap[lf_idx,j,0])
                ind_pt += 1
                if ind_pt == ind_len:
                    indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                    ind_len += extend_mem
        else:
            for j in range(mmap.shape[1]):
                k = 0
                isin = 0
                while (k < mmap.shape[2]-1) and (isin == k):
                    if j == mmap.shape[1]-1:
                        if mmap[lf_idx,j,0] == -1.:
                            k += 1
                            continue
                    leaf_val = mmap[lf_idx,j,k+1]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = int(mmap[lf_idx,j,0])
                    ind_pt += 1

                    if ind_pt == ind_len:
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit_leaves(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_leaves,loaded_leaves,1)
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit_leaves(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_leaves,loaded_leaves,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit_leaves(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_leaves,loaded_leaves,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit_leaves(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_leaves,loaded_leaves,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit_leaves(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_leaves,loaded_leaves,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_limit_leaves(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_leaves,loaded_leaves,0)
            
    return indices,ind_pt,ind_len,loaded_leaves

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int check_intersect(float[:,:] bounds,float[:] mins,float[:] maxs) nogil:
    cdef int intersects, idx
    
    intersects = 0
    idx = 0
    while (idx < bounds.shape[0]) and (intersects == idx):
        if (bounds[idx,1] >= mins[idx]) and (bounds[idx,0] <= maxs[idx]):
            intersects += 1
        idx += 1
    
    if intersects == idx:
        intersects = 1
    else:
        intersects = 0
    
    return intersects

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int check_contained(float[:,:] bounds,float[:] mins,float[:] maxs) nogil:
    cdef int contained, idx
    
    contained = 0
    idx = 0
    while (idx < bounds.shape[0]) and (contained == idx):
        if (bounds[idx,0] >= mins[idx]) and (bounds[idx,1] <= maxs[idx]):
            contained += 1
        idx += 1
        
    if contained == idx:
        contained = 1
    else:
        contained = 0
    
    return contained


#TODO so far contains both solutions malloc and realloc -> remove one of them
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef long* resize_long_array(long* arr,long old_len, long new_len) nogil:
    cdef long i 
    cdef long* mem = <long*> realloc(arr,new_len * sizeof(long))
    #cdef long* mem = <long*> malloc(new_len * sizeof(long))
    #if not mem:
    #    raise MemoryError()
    #for i in range(old_len):
    #    mem[i] = arr[i]
    arr = mem
    return arr
