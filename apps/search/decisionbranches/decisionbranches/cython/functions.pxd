cpdef double search_boxes(int , double[::1,:],  int[::1] ,int[::1], int , int , int[::1]  , int ,
                          int ,int, int,double , int, double[:,::1] , int[::1],
                          double[::1] ,double[::1] ,int[::1] ,int[::1],double [:,::1], int[::1],int[::1] ) 

cdef int create_box(double[:,::1] ,int[::1] ,double [::1,:] , int[::1] ,int[::1] , int ,int ,int ,int[::1] ,
                     int , int ,int ,double ,int , double[::1] , double[::1] , int* ,int* , double* ,double* ) nogil


cdef void expand(double [:,::1] , int[::1] , double [::1,:] , int[::1] ,int[::1] , int , int , int[::1] , int ,
                int ,int , double , int , int[::1] , int[::1] ,int[::1] ,double[::1] ,double[::1] ,
                int* , int* , double* ,double* ) nogil

cdef double find_split(int[::1],double[::1,:] ,int[::1] ,int ,int[::1] , int ,double[::1] ,
                        int , double , int , int ,int[::1] ,int ,int ,int , int , double , int ,
                       int[::1] , int* , int* , double* ,double* ) nogil

cdef double gini_impurity(int* ,int* , int[::1] ) nogil

cdef double find_closest_rare_overall(double [::1,:] ,int [::1] ,int ,int ,int,int,int ,double ,double*,double*) nogil

cdef double find_furthest_rare_overall(double [::1,:] ,int [::1] ,double ,int ,int,int,int ,double,double*,double* ) nogil

#cdef double find_closest_rare_overall_box(double [::1,:] ,int [::1] ,double ,int ,int,int,int ,double ) nogil

cdef int filter_all(double[:,::1] ,  double[::1, :] , int[::1] ) nogil

cdef int filter_all_red(double[:,::1] ,  double[::1, :] , int[::1] ,int* ,int ) nogil

cpdef int filter_py(double[:,::1] ,  double[::1, :] , int[::1] )

cdef int check_complete_duplicates(int[::1] ,int ,int[::1] , int[::1] ,int*,int*) nogil
