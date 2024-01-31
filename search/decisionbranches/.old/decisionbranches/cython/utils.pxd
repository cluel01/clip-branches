cdef int sum_iarr(int[::1]) nogil

cdef int sum_iarr_pt(int* ,int ) nogil

cdef void find_extreme_values(double[::1,:],double* ,double* ) nogil

cdef double split (int, int ) nogil

cdef int sort_unique(double[::1] ,int , int ) nogil

cdef void shuffle_array(int[::1]) nogil
