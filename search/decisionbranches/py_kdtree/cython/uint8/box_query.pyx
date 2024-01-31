from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdlib cimport malloc, free, realloc
from libc.stdint cimport uint32_t
        
cimport cython
#from cython.parallel import prange
import numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef uint32_t[::1] recursive_search_box_uint32(unsigned char[::1] mins, unsigned char[::1] maxs, unsigned char[:,:,::1] tree,const uint32_t[:,::1] leaf_index_map, 
                                    int n_leaves, int n_nodes, const unsigned char[:,:,::1] mmap, 
                                    int max_pts, int max_leaves, double mem_cap, int[::1] arr_loaded):    
    cdef uint32_t[::1] indices_view
    cdef uint32_t ind_len = int(mmap.shape[0] * mmap.shape[1] * mem_cap) 
    cdef uint32_t extend_mem = ind_len

    cdef uint32_t ind_pt = 0 
    cdef uint32_t* indices = <uint32_t*> malloc(ind_len * sizeof(uint32_t))

    cdef int loaded_leaves = 0

    try:
        indices, ind_pt, ind_len, loaded_leaves = _recursive_search_box_uint32(0, mins, maxs, tree,leaf_index_map, n_leaves, n_nodes, indices, ind_pt, ind_len, mmap, extend_mem, loaded_leaves, 0)
        arr_loaded[0] = loaded_leaves
        indices_view = np.empty(ind_pt, dtype=np.uint32)
        for i in range(ind_pt):
            indices_view[i] = indices[i]
        return indices_view 
    finally:
        free(indices)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (uint32_t*, uint32_t, uint32_t, int) _recursive_search_box_uint32(int node_idx, unsigned char[::1] mins, unsigned char[::1] maxs, 
                                                    unsigned char[:,:,::1] tree,const uint32_t[:,::1] leaf_index_map, int n_leaves, int n_nodes,
                                                    uint32_t* indices, uint32_t ind_pt, uint32_t ind_len, 
                                                    const unsigned char[:,:,::1] mmap, uint32_t extend_mem, 
                                                    int loaded_leaves, int contained) nogil:
    cdef int l_idx, r_idx, intersects, ret, lf_idx, isin, j, k
    l_idx, r_idx = (2 * node_idx) + 1, (2 * node_idx) + 2
    cdef unsigned char[:,:] bounds, l_bounds, r_bounds
    cdef unsigned char leaf_val
    cdef uint32_t MAX_INT = 4294967295

    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves + node_idx - n_nodes
        loaded_leaves += 1
        if contained == 1:
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if leaf_index_map[lf_idx,j] == MAX_INT:
                        continue
                indices[ind_pt] = leaf_index_map[lf_idx,j]
                ind_pt += 1

                if ind_pt == ind_len:
                    indices = resize_uint32_array(indices,ind_len,ind_len+extend_mem)
                    ind_len += extend_mem
        else:
            for j in range(mmap.shape[1]):
                k = 0
                isin = 0
                while (k < mmap.shape[2]) and (isin == k):
                    if j == mmap.shape[1]-1:
                        if leaf_index_map[lf_idx,j] == MAX_INT:
                            k += 1
                            continue
                    leaf_val = mmap[lf_idx,j,k]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = leaf_index_map[lf_idx,j]
                    ind_pt += 1
                    if ind_pt == ind_len:
                        indices = resize_uint32_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices, ind_pt, ind_len, loaded_leaves
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_uint32(l_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_uint32(r_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_uint32(l_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_uint32(l_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_uint32(r_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_uint32(r_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)

    return indices, ind_pt, ind_len, loaded_leaves


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef long[::1] recursive_search_box_int64(unsigned char[::1] mins, unsigned char[::1] maxs, unsigned char[:,:,::1] tree,const long[:,::1] leaf_index_map, 
                                    int n_leaves, int n_nodes, const unsigned char[:,:,::1] mmap, 
                                    int max_pts, int max_leaves, double mem_cap, int[::1] arr_loaded):    
    cdef long[::1] indices_view
    cdef long ind_len = int(mmap.shape[0] * mmap.shape[1] * mem_cap) 
    cdef long extend_mem = ind_len

    cdef long ind_pt = 0 
    cdef long* indices = <long*> malloc(ind_len * sizeof(long))

    cdef int loaded_leaves = 0

    try:
        indices, ind_pt, ind_len, loaded_leaves = _recursive_search_box_int64(0, mins, maxs, tree,leaf_index_map, n_leaves, n_nodes, indices, ind_pt, ind_len, mmap, extend_mem, loaded_leaves, 0)
        arr_loaded[0] = loaded_leaves
        indices_view = np.empty(ind_pt, dtype=np.int64)
        for i in range(ind_pt):
            indices_view[i] = indices[i]
        return indices_view 
    finally:
        free(indices)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long*, long, long, int) _recursive_search_box_int64(int node_idx, unsigned char[::1] mins, unsigned char[::1] maxs, 
                                                    unsigned char[:,:,::1] tree,const long[:,::1] leaf_index_map, int n_leaves, int n_nodes,
                                                    long* indices, long ind_pt, long ind_len, 
                                                    const unsigned char[:,:,::1] mmap, long extend_mem, 
                                                    int loaded_leaves, int contained) nogil:
    cdef int l_idx, r_idx, intersects, ret, lf_idx, isin, j, k
    l_idx, r_idx = (2 * node_idx) + 1, (2 * node_idx) + 2
    cdef unsigned char[:,:] bounds, l_bounds, r_bounds
    cdef unsigned char leaf_val
    cdef long MAX_INT = 9223372036854775807

    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves + node_idx - n_nodes
        loaded_leaves += 1
        if contained == 1:
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if leaf_index_map[lf_idx,j] == MAX_INT:
                        continue
                indices[ind_pt] = leaf_index_map[lf_idx,j]
                ind_pt += 1

                if ind_pt == ind_len:
                    indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                    ind_len += extend_mem
        else:
            for j in range(mmap.shape[1]):
                k = 0
                isin = 0
                while (k < mmap.shape[2]) and (isin == k):
                    if j == mmap.shape[1]-1:
                        if leaf_index_map[lf_idx,j] == MAX_INT:
                            k += 1
                            continue
                    leaf_val = mmap[lf_idx,j,k]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = leaf_index_map[lf_idx,j]
                    ind_pt += 1
                    if ind_pt == ind_len:
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices, ind_pt, ind_len, loaded_leaves
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_int64(l_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_int64(r_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_int64(l_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_int64(l_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_int64(r_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_box_int64(r_idx,mins,maxs,tree,leaf_index_map,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)

    return indices, ind_pt, ind_len, loaded_leaves

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef int check_intersect(unsigned char[:,:] bounds, unsigned char[:] mins, unsigned char[:] maxs) nogil:
    cdef int intersects, idx
    
    intersects = 0
    idx = 0
    while (idx < bounds.shape[0]) and (intersects == idx):
        if (bounds[idx, 1] >= mins[idx]) and (bounds[idx, 0] <= maxs[idx]):
            intersects += 1
        idx += 1
    
    return 1 if intersects == idx else 0

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef int check_contained(unsigned char[:,:] bounds, unsigned char[:] mins, unsigned char[:] maxs) nogil:
    cdef int contained, idx
    
    contained = 0
    idx = 0
    while (idx < bounds.shape[0]) and (contained == idx):
        if (bounds[idx, 0] >= mins[idx]) and (bounds[idx, 1] <= maxs[idx]):
            contained += 1
        idx += 1
    
    return 1 if contained == idx else 0

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef long* resize_long_array(long* arr, long old_len, long new_len) nogil:
    cdef long* mem = <long*> realloc(arr, new_len * sizeof(long))
    return mem

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef uint32_t* resize_uint32_array(uint32_t* arr, uint32_t old_len, uint32_t new_len) nogil:
    cdef uint32_t* mem = <uint32_t*> realloc(arr, new_len * sizeof(uint32_t))
    return mem
