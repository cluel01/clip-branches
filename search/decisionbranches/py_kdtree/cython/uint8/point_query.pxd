from libc.stdint cimport uint8_t, int64_t, uint32_t
from libc.math cimport double

cimport cython

cpdef int recursive_search_point_int64(uint8_t[::1] point, int k, uint8_t[:,:,::1] tree, const long[:,::1] leaf_index_map, 
                                 int n_leaves, int n_nodes, int stop_leaves, const uint8_t[:,:,::1] mmap,
                                 int64_t[::1] indices_view, double[::1] distances_view)

cpdef int recursive_search_point_uint32(uint8_t[::1] point, int k, uint8_t[:,:,::1] tree, const uint32_t[:,::1] leaf_index_map, 
                                 int n_leaves, int n_nodes, int stop_leaves, const uint8_t[:,:,::1] mmap,
                                 uint32_t[::1] indices_view, double[::1] distances_view)
                            

