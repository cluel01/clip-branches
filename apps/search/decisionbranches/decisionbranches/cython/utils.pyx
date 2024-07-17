from cython.parallel import prange
cimport cython
from libc.stdlib cimport malloc, free, rand, RAND_MAX, srand,qsort

cdef extern from "math.h":
    float INFINITY

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int sum_iarr(int[::1] arr) nogil:
    cdef int count = 0  
    cdef int i
    for i in prange(arr.shape[0],nogil=True):
        count += arr[i]
    return count

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int sum_iarr_pt(int* arr,int L) nogil:
    cdef int count = 0  
    cdef int i
    for i in prange(L,nogil=True):
        count += arr[i]
    return count

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void find_extreme_values(double[::1,:] x,double* x_max,double* x_min) nogil:
    cdef int i,j
    cdef double MAX,MIN
    for i in range(x.shape[1]):
        MAX = -INFINITY
        MIN = INFINITY
        for j in range(x.shape[0]):
            if x[j,i] > MAX:
                MAX = x[j,i]
            elif x[j,i] < MIN:
                MIN = x[j,i]
        x_max[i] = MAX
        x_min[i] = MIN



@cython.cdivision(True) 
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double generate_double_random(double lower, double upper, int seed) nogil:
    '''Based on https://stackoverflow.com/a/33059025'''
    #srand(seed)
    cdef double number
    cdef double drange = upper - lower 
    cdef double div
    div = RAND_MAX/drange
    number = lower + (rand() / div)
    return number


#ctypedef enum SplitType: HALF=0, MAX=1, RANDOM=2, MIN=3
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double split(int split_type, int seed) nogil:
    if split_type == 0:
        return 0.5
    elif split_type == 1:
        return 0.9999
    elif split_type == 2:
        return generate_double_random(0.0, 0.9999, seed)
    elif split_type == 3:
        return 0.0001
    else:
        return -1 # Error      


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
@cython.nonecheck(False)
cdef int asc(const void* a, const void* b) nogil:
    cdef double a_v = (<double*>a)[0]
    cdef double b_v = (<double*>b)[0]
    if a_v < b_v:
        return -1
    elif a_v == b_v:
        return 0
    else:
        return 1

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
@cython.nonecheck(False)
cdef int desc(const void* a, const void* b) nogil:
    cdef double a_v = (<double*>a)[0]
    cdef double b_v = (<double*>b)[0]
    if a_v < b_v:
        return 1
    elif a_v == b_v:
        return 0
    else:
        return -1

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
@cython.nonecheck(False)
cdef int unique(double[::1] arr,int pointer) nogil:
    cdef int i
    cdef double* tmp = <double*> malloc(pointer * sizeof(double))
    cdef int j = 0    
    
    try:
        if pointer < 2:
            return pointer # only one point 

        for i in range(pointer-1):
            if arr[i] != arr[i+1]:
                tmp[j] = arr[i]
                j += 1

        tmp[j] = arr[pointer-1]
        j += 1

        for i in range(j):
            arr[i] = tmp[i]
            
        return j
    finally:
        free(tmp)

        
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
@cython.nonecheck(False)
#order == 0 asc | order == 1 desc
cdef int sort_unique(double[::1] arr,int pointer, int order) nogil:
    cdef int last
    if order == 0:
        # a needn't be C continuous because strides helps
        qsort(&arr[0], pointer, arr.strides[0], &asc)
    if order == 1:
        # a needn't be C continuous because strides helps
        qsort(&arr[0], pointer, arr.strides[0], &desc)

    last = unique(arr,pointer)
    return last

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void shuffle_array(int[::1] array) nogil:
    cdef int i
    cdef int j
    cdef int t
    cdef int n = array.shape[0]
    if (n > 1):
        for i in range(n):
            j = i + rand() / (RAND_MAX / (n - i) + 1)
            t = array[j]
            array[j] = array[i]
            array[i] = t
