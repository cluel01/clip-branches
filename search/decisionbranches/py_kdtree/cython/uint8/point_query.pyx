# cython: profile=False
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport fabs,sqrt
from libc.stdint cimport uint32_t

cimport cython

cdef extern from "math.h":
    double INFINITY

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef int recursive_search_point_uint32(unsigned char[::1] point, int k, unsigned char[:,:,::1] tree,const uint32_t[:,::1] leaf_index_map, int n_leaves,
                                 int n_nodes, int stop_leaves, const unsigned char[:,:,::1] mmap,
                                 uint32_t[::1] indices_view, double[::1] distances_view):
    cdef int i
    cdef int leaf_count = 0
    cdef long depth = 0
    cdef uint32_t* indices = <uint32_t*> malloc(k * sizeof(uint32_t))
    cdef double* distances = <double*> malloc(k * sizeof(double))

    try:
        # initializes distances
        for i in range(k):
            distances[i] = INFINITY

        indices, distances, leaf_count = _recursive_search_point_uint32(0, point, k, depth, tree,leaf_index_map, n_leaves, n_nodes, stop_leaves, leaf_count, indices, distances, mmap)
        for i in range(k):
            indices_view[i] = indices[i]
            distances_view[i] = distances[i]
        return leaf_count
    finally:
        free(indices)
        free(distances)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef (uint32_t*, double*, int) _recursive_search_point_uint32(int node_idx, unsigned char[::1] point, int k, int depth, unsigned char[:,:,::1] tree,const uint32_t[:,::1] leaf_index_map, int n_leaves, int n_nodes,
                                                   int stop_leaves, int leaf_count, uint32_t* indices, double* distances, const unsigned char[:,:,::1] mmap) nogil:
    cdef int l_idx, r_idx, axis, first, second, lf_idx, dist_idx, i, j
    l_idx, r_idx = (2 * node_idx) + 1, (2 * node_idx) + 2
    cdef double[:,:] l_bound, r_bound
    cdef double v
    cdef double median, max_dist, max_dist_sub, dist, sub, power
    cdef uint32_t MAX_INT = 4294967295

    if leaf_count >= stop_leaves:
        return indices, distances, leaf_count

    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves + node_idx - n_nodes
        for j in range(mmap.shape[1]):
            if j == mmap.shape[1] - 1:
                if leaf_index_map[lf_idx,j] == MAX_INT:
                    continue
            dist = 0
            for i in range(tree.shape[1]):  # dimensionality
                v = <double>mmap[lf_idx, j, i ]
                sub =  v - point[i]
                power = sub * sub
                dist += power
            dist = sqrt(dist)
            max_dist = get_max(distances, k)
            if dist < max_dist:
                dist_idx = get_max_idx(distances, k)
                distances[dist_idx] = dist
                indices[dist_idx] = leaf_index_map[lf_idx,j] #int(mmap[lf_idx, j, 0])
        leaf_count += 1
        return indices, distances, leaf_count

    ############################## Internal Node ################################################################
    else:
        axis = depth % tree.shape[1]
        median = tree[l_idx][axis][1]
        if point[axis] < median:
            first = l_idx
            second = r_idx
        else:
            first = r_idx
            second = l_idx
        indices, distances, leaf_count = _recursive_search_point_uint32(first, point, k, depth + 1, tree,leaf_index_map, n_leaves, n_nodes, stop_leaves, leaf_count, indices, distances, mmap)
        max_dist = get_max(distances, k)
        max_dist_sub = fabs(median - point[axis])
        if max_dist_sub < max_dist:
            indices, distances, leaf_count = _recursive_search_point_uint32(second, point, k, depth + 1, tree,leaf_index_map, n_leaves, n_nodes, stop_leaves, leaf_count, indices, distances, mmap)

        return indices, distances, leaf_count


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef int recursive_search_point_int64(unsigned char[::1] point, int k, unsigned char[:,:,::1] tree,const long[:,::1] leaf_index_map, int n_leaves,
                                 int n_nodes, int stop_leaves, const unsigned char[:,:,::1] mmap,
                                 long[::1] indices_view, double[::1] distances_view):
    cdef int i
    cdef int leaf_count = 0
    cdef long depth = 0
    cdef long* indices = <long*> malloc(k * sizeof(long))
    cdef double* distances = <double*> malloc(k * sizeof(double))

    try:
        # initializes distances
        for i in range(k):
            distances[i] = INFINITY

        indices, distances, leaf_count = _recursive_search_point_int64(0, point, k, depth, tree,leaf_index_map, n_leaves, n_nodes, stop_leaves, leaf_count, indices, distances, mmap)
        for i in range(k):
            indices_view[i] = indices[i]
            distances_view[i] = distances[i]
        return leaf_count
    finally:
        free(indices)
        free(distances)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef (long*, double*, int) _recursive_search_point_int64(int node_idx, unsigned char[::1] point, int k, int depth, unsigned char[:,:,::1] tree,const long[:,::1] leaf_index_map, int n_leaves, int n_nodes,
                                                   int stop_leaves, int leaf_count, long* indices, double* distances, const unsigned char[:,:,::1] mmap) nogil:
    cdef int l_idx, r_idx, axis, first, second, lf_idx, dist_idx, i, j
    l_idx, r_idx = (2 * node_idx) + 1, (2 * node_idx) + 2
    cdef double[:,:] l_bound, r_bound
    cdef double v
    cdef double median, max_dist, max_dist_sub, dist, sub, power
    cdef long MAX_INT = 9223372036854775807

    if leaf_count >= stop_leaves:
        return indices, distances, leaf_count

    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves + node_idx - n_nodes
        for j in range(mmap.shape[1]):
            if j == mmap.shape[1] - 1:
                if leaf_index_map[lf_idx,j] == MAX_INT:
                    continue
            dist = 0
            for i in range(tree.shape[1]):  # dimensionality
                v = <double>mmap[lf_idx, j, i ]
                sub =  v - point[i]
                power = sub * sub
                dist += power
            dist = sqrt(dist)
            max_dist = get_max(distances, k)
            if dist < max_dist:
                dist_idx = get_max_idx(distances, k)
                distances[dist_idx] = dist
                indices[dist_idx] = leaf_index_map[lf_idx,j] #int(mmap[lf_idx, j, 0])
        leaf_count += 1
        return indices, distances, leaf_count

    ############################## Internal Node ################################################################
    else:
        axis = depth % tree.shape[1]
        median = tree[l_idx][axis][1]
        if point[axis] < median:
            first = l_idx
            second = r_idx
        else:
            first = r_idx
            second = l_idx
        indices, distances, leaf_count = _recursive_search_point_int64(first, point, k, depth + 1, tree,leaf_index_map, n_leaves, n_nodes, stop_leaves, leaf_count, indices, distances, mmap)
        max_dist = get_max(distances, k)
        max_dist_sub = fabs(median - point[axis])
        if max_dist_sub < max_dist:
            indices, distances, leaf_count = _recursive_search_point_int64(second, point, k, depth + 1, tree,leaf_index_map, n_leaves, n_nodes, stop_leaves, leaf_count, indices, distances, mmap)

        return indices, distances, leaf_count



@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double get_max(double* array,int size) nogil:
    cdef int i
    cdef double MAX = -INFINITY
    for i in range(size):
        if array[i] > MAX:
            MAX = array[i]
    return MAX

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef int get_max_idx(double* array,int size) nogil:
    cdef int i
    cdef double MAX = -INFINITY
    cdef int idx
    for i in range(size):
        if array[i] > MAX:
            MAX = array[i]
            idx = i
    return idx



    
    
