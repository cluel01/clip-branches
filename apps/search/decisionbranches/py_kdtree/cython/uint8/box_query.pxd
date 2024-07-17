# my_module.pxd
cimport cython
from libc.stdint cimport uint8_t, int64_t, uint32_t

cpdef int64_t[::1] recursive_search_box_int64(uint8_t[::1] mins, uint8_t[::1] maxs, uint8_t[:,:,::1] tree, const long[:,::1] leaf_index_map, 
                                       int n_leaves, int n_nodes, const uint8_t[:,:,::1] mmap, 
                                       int max_pts, int max_leaves, double mem_cap, int[::1] arr_loaded)

cpdef uint32_t[::1] recursive_search_box_uint32(uint8_t[::1] mins, uint8_t[::1] maxs, uint8_t[:,:,::1] tree, const uint32_t[:,::1] leaf_index_map, 
                                       int n_leaves, int n_nodes, const uint8_t[:,:,::1] mmap, 
                                       int max_pts, int max_leaves, double mem_cap, int[::1] arr_loaded)