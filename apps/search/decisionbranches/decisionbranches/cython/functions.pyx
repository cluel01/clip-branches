import numpy as np
cimport numpy as cnp


cimport cython
from libc.math cimport pow as cpow,fabs
from libc.stdlib cimport malloc, free, srand, qsort

from decisionbranches.cython.utils cimport sum_iarr,sum_iarr_pt,sort_unique,split,shuffle_array,find_extreme_values#,asc,desc

cdef extern from "math.h":
    float INFINITY

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cpdef double search_boxes(int pt_idx, double[::1,:] x,  int[::1] y, int[::1] feats, int min_pts, int early_stopping, int[::1] class_weights, int splitter,
                          int min_nobj,int stop_infinite,int stop_infinite_split,double eps, int seed, double[:,::1] best_bbox, int[::1] best_pts,
                          double[::1] min_vals,double[::1] max_vals,int[::1] min_mask,int[::1] max_mask,double [:,::1] bbox, int[::1] pts_idx,int[::1] filter_pts_idx):
    srand(seed) # Seed random numbers
    cdef int j,nobj,ret
    
    #array counting how many rare and non rare are inside and outside the box
    cdef int* y_box_in = <int*> malloc(
        2 * sizeof(int))
    cdef int* y_box_out = <int*> malloc(
        2 * sizeof(int))
    
    cdef int nrare = sum_iarr(y)

    cdef double score = INFINITY

    cdef double* x_max = <double*> malloc(
        x.shape[1] * sizeof(double))
    cdef double* x_min = <double*> malloc(
        x.shape[1] * sizeof(double))

    try:
        y_box_in[1] = 1
        y_box_in[0] = 0
        y_box_out[1] = nrare-1 
        y_box_out[0] = x.shape[0]-nrare

        find_extreme_values(x,x_max,x_min)

        ret = create_box(bbox,pts_idx,x, y, feats, pt_idx, min_pts, early_stopping, class_weights,
            splitter, stop_infinite,stop_infinite_split,eps, seed,min_vals,max_vals,y_box_in,y_box_out,x_max,x_min) # score and pts_idx are passed here to be returned by the function

        if ret == 0:
            expand(bbox,pts_idx,x,y,feats,min_pts,early_stopping,class_weights,splitter,stop_infinite,stop_infinite_split,eps,seed,filter_pts_idx,min_mask,max_mask,min_vals,max_vals,y_box_in,y_box_out,x_max,x_min)

            ##### Temporary fix #######
            nobj = filter_all(bbox,x,best_pts)

            y_box_in[1] = 0
            y_box_in[0] = 0
            y_box_out[1] = 0 
            y_box_out[0] = 0

            for i in range(len(y)):
                if best_pts[i] == 1:
                    if y[i] == 1:
                        y_box_in[1] += 1
                    else:
                        y_box_in[0] += 1
                else:
                    if y[i] == 1:
                        y_box_out[1] += 1
                    else:
                        y_box_out[0] += 1

            ###########################

            if nobj >= min_nobj:

                score = gini_impurity(y_box_in,y_box_out,class_weights)

                #copy bbox
                for j in range(bbox.shape[0]):
                    best_bbox[j] = bbox[j]
        return score
    finally:
        free(y_box_in)
        free(y_box_out)
        free(x_max)
        free(x_min)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef int create_box(double[:,::1] bbox,int[::1] pts_idx,double [::1,:] x, int[::1] y,int[::1] feats, int pt_idx,int min_pts,int early_stopping,int[::1] cw,
                     int splitter, int stop_infinite,int stop_infinite_split,double eps,int seed, double[::1] min_vals, double[::1] max_vals, int* y_box_in,int* y_box_out, double* x_max,double* x_min) nogil:
    cdef int i,j,f
    cdef int min_pt,max_pt,eq_count,ret
    cdef int allp = 0
    cdef double outer, inner,border
    
    cdef double closest_min,closest_max
    
    #Set pt_idx to false
    pts_idx[pt_idx] = 0
    
    for j in range(feats.shape[0]):
        f = feats[j]
        min_pt,max_pt,eq_count = 0,0,0

        closest_min = -INFINITY
        closest_max = INFINITY
        
        if allp == 0:
            for i in range(pts_idx.shape[0]):
                # if either pts_idx is still true or it is the first round and not the center point to reset the idxs 
                if (pts_idx[i] == 1) or ((j == 0) and (i != pt_idx)):
                    if x[i,f] > x[pt_idx,f]:
                        #max_vals[max_pt] = x[i,f]
                        max_pt += 1
                        pts_idx[i] = 0
                        if x[i,f] < closest_max:
                            closest_max = x[i,f]
                    elif x[i,f] < x[pt_idx,f]:
                        #min_vals[min_pt] = x[i,f]
                        min_pt += 1
                        pts_idx[i] = 0
                        if x[i,f] > closest_min:
                            closest_min = x[i,f]
                    else:
                        eq_count += 1
                        pts_idx[i] = 1


        if max_pt > 0:
            outer = closest_max
            border = x[pt_idx,f] + (outer-x[pt_idx,f]) * split(1,seed) # max according to MER behaviour (only comes into play for first dimension or following dimension in case of duplicate values)
        else:
            if f > 0:
                border = find_closest_rare_overall(x,y,pt_idx,f,1,3,seed,eps,x_max,x_min) # min -> is expanded anyway afterwards -> before that: previous dimensions expand
            else:
                if stop_infinite == 1:
                    outer = x_max[f]
                    border = x[pt_idx,f] + (outer-x[pt_idx,f]) * split(stop_infinite_split,seed) #random split
                else:
                    border = INFINITY
        bbox[f,1] = border
            
        if min_pt > 0:
            outer = closest_min
            border = x[pt_idx,f] + (outer-x[pt_idx,f]) * split(1,seed) # max according to MER behaviour
        else:
            if f > 0:
                border = find_closest_rare_overall(x,y,pt_idx,f,-1,3,seed,eps,x_max,x_min) ## min -> is expanded anyway afterwards -> before that: previous dimensions expand
            else:
                if stop_infinite == 1:
                    outer = x_min[f] 
                    border = x[pt_idx,f] + (outer-x[pt_idx,f]) * split(stop_infinite_split,seed) #random split
                else:
                    border = -1*INFINITY
        bbox[f,0] = border            
        
        #in case there are no points to be compared -> no further scanning required
        if allp == 0:
            if eq_count == 0:
                allp = 1
                
    #if this is still 0 -> we have a complete duplicate
    if allp == 0:
        ret = check_complete_duplicates(y,pt_idx,cw,pts_idx,y_box_in,y_box_out)
        if ret == 1:
            pts_idx[pt_idx] = 1
            return 1

    pts_idx[pt_idx] = 1
    return 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
@cython.nonecheck(False)
cdef void expand(double [:,::1] bbox, int[::1] box_pts, double [::1,:] x, int[::1] y,int[::1] feats, int min_pts, int early_stopping, int[::1] cw, int splitter,
                int stop_infinite,int stop_infinite_split, double eps, int seed, int[::1] pts_idx, int[::1] min_mask,int[::1] max_mask,double[::1] min_vals,double[::1] max_vals,
                int* y_box_in, int* y_box_out, double* x_max,double* x_min) nogil:
    cdef int i,j,k
    cdef int f
    cdef int isum, p_feats, p_mins,p_maxs,ncount
    
    cdef double minX,maxX,min_border, max_border
    
    cdef int p_vals, limit, split_idx
    cdef int n_rare = 0   
    
    cdef int size = feats.shape[0]-1
    cdef int* feat_filter = <int*> malloc(size * sizeof(int))
    
    try:
        for i in range(feats.shape[0]):
            f = feats[i]
            #fill feat_filter
            p_feats = 0
            for j in range(feats.shape[0]):                
                if feats[j] != f:
                    feat_filter[p_feats] = feats[j]
                    p_feats += 1      
            
            #filter_red(bbox,x,pts_idx,feat_filter,size) 
            isum = filter_all_red(bbox,x,pts_idx,feat_filter,size) 

            if isum > 0:
                minX = bbox[f,0]
                maxX = bbox[f,1]
                
                p_maxs = 0
                p_mins = 0
                for k in range(pts_idx.shape[0]):
                    min_mask[k] = 0
                    max_mask[k] = 0
                    if (pts_idx[k] == 1) and (box_pts[k] == 0):
                        if x[k,f] > maxX:
                            #max_mask[k] = 1
                            max_mask[p_maxs] = k
                            max_vals[p_maxs] = x[k,f]
                            p_maxs += 1
                            
                        if x[k,f] < minX:
                            #min_mask[k] = 1
                            min_mask[p_mins] = k
                            min_vals[p_mins] = x[k,f]
                            p_mins += 1
                
                min_border = find_split(box_pts,x,y,f,min_mask,
                                        min_pts,min_vals,p_mins,minX,-1,early_stopping,cw,splitter,stop_infinite,stop_infinite_split,feats.shape[0]-i,eps,seed,pts_idx,y_box_in,y_box_out,x_max,x_min)

                #input pts_idx as memory area for storing best_box_pts of find_split method
                max_border = find_split(box_pts,x,y,f,max_mask,
                                        min_pts,max_vals,p_maxs,maxX,1,early_stopping,cw,splitter,stop_infinite,stop_infinite_split,feats.shape[0]-i,eps,seed,pts_idx,y_box_in,y_box_out,x_max,x_min)
            else:
                min_border = -1*INFINITY
                max_border = INFINITY

            if stop_infinite == 1:
                if min_border == -1*INFINITY:
                    #Last feat
                    if i == feats.shape[0]-1:
                        #min_border = bbox[f,0] - fabs(bbox[f,0] * split(splitter,seed))
                        min_border = bbox[f,0] - fabs((x_min[f] - bbox[f,0]) * split(stop_infinite_split,seed)) #Random split
                    else:
                        min_border = find_furthest_rare_overall(x,y,bbox[f,0],f,-1,stop_infinite_split,seed,eps,x_max,x_min) #random split #TODO here even decrease the random range?

                if max_border == INFINITY:
                    #Last feat
                    if i == feats.shape[0]-1:
                        #max_border = bbox[f,1] + fabs(bbox[f,1] * split(splitter,seed))
                        max_border = bbox[f,1] + fabs((x_max[f] - bbox[f,1]) * split(2,seed)) # random split
                    else:
                        max_border = find_furthest_rare_overall(x,y,bbox[f,1],f,1,stop_infinite_split,seed,eps,x_max,x_min) #random split TODO here even decrease the random range? we already searched for the furthest rare anyway!

            bbox[f,0] = min_border
            bbox[f,1] = max_border
    finally:
        free(feat_filter)

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double find_split(int[::1] box_pts,double[::1,:] x,int[::1] y,int f,int[::1] mask, int min_pts,double[::1] lim_values,
                        int pt_lim_vals, double lim, int c, int early_stopping,int[::1] cw,int splitter,int stop_infinite,int stop_infinite_split, int nfeat, double eps, int seed,
                       int[::1] best_box_pts, int* y_box_in, int* y_box_out, double* x_max,double* x_min) nogil:
    cdef int i,j,k,limit,p_vals,idx
    cdef int pointer = 0
    cdef int n_rare
    cdef int split_idx = -1
    cdef double score, best_score,outer,inner,val,border
    cdef double last_lim_val = c*-INFINITY
    
    cdef int* best_y_box_in = <int*> malloc(2 * sizeof(int))
    cdef int* best_y_box_out = <int*> malloc(2 * sizeof(int))
    
    try:
        best_y_box_in[0] = y_box_in[0]
        best_y_box_in[1] = y_box_in[1]
        best_y_box_out[0] = y_box_out[0]
        best_y_box_out[1] = y_box_out[1]


        if pt_lim_vals > 0:
            if c == 1:
                p_vals = sort_unique(lim_values,pt_lim_vals,0)
            else:
                p_vals = sort_unique(lim_values,pt_lim_vals,1)

            limit = min_pts
            if p_vals < min_pts:
                limit = p_vals

            #early stopping?
            n_rare = 0

            if early_stopping == 1:
                for i in range(pt_lim_vals):
                    idx = mask[i]
                    if y[idx] == 1:
                        if c*x[idx,f] <= c*lim_values[p_vals-1]:
                            n_rare += 1

            if (early_stopping == 0) or ((early_stopping == 1) and (n_rare>0)):
                #copy original box_ptx to best_box in case no better is found
                for i in range(box_pts.shape[0]):
                    best_box_pts[i] = box_pts[i]

                best_score = gini_impurity(y_box_in,y_box_out,cw)

                for i in range(limit):
                    for j in range(pt_lim_vals):
                        idx = mask[j]
                        if c*last_lim_val < c*x[idx,f] <= c*lim_values[i]:
                            if box_pts[idx] == 0:
                                box_pts[idx] = 1

                                if y[idx] == 1:
                                    y_box_in[1] += 1
                                    y_box_out[1] -= 1
                                else:
                                    y_box_in[0] += 1
                                    y_box_out[0] -= 1
                    last_lim_val = lim_values[i]

                    score = gini_impurity(y_box_in,y_box_out,cw)
                    
                    if score < best_score:
                        best_score = score
                        split_idx = i
                        #copy box
                        for k in range(box_pts.shape[0]):
                            best_box_pts[k] = box_pts[k]
                        #copy count    
                        best_y_box_in[0] = y_box_in[0]
                        best_y_box_in[1] = y_box_in[1]
                        best_y_box_out[0] = y_box_out[0]
                        best_y_box_out[1] = y_box_out[1]

                ##### End - find split #####
                for i in range(box_pts.shape[0]):
                    box_pts[i] = best_box_pts[i]

                y_box_in[0] = best_y_box_in[0]
                y_box_in[1] = best_y_box_in[1]
                y_box_out[0] = best_y_box_out[0]
                y_box_out[1] = best_y_box_out[1]

            #set outer inner value
            if split_idx == -1:
                outer = lim_values[0]
                inner = lim
            else:
                inner = lim_values[split_idx]
                if p_vals > split_idx+1:
                    outer = lim_values[split_idx+1]
                else:
                    if stop_infinite == 1:
                        #Here the larger the better the expansion
                        #outer = lim_values[split_idx] #  only this would lead to inner == outer

                        #Last feat
                        if nfeat == 1:
                            if c == 1:
                                outer = x_max[f]
                            else:
                                outer = x_min[f]
                            border = inner + (outer-inner) * split(2,seed) #random split
                        else:                            
                            border = find_furthest_rare_overall(x,y,inner,f,c,2,seed,eps,x_max,x_min) #random split
                        return border
                    else:
                        return c*INFINITY

            #set border 
            if outer == inner:
                val = inner * 0.0001
                if val == 0:
                    val = eps
                border = inner + c*fabs(val)
            else:
                border = inner + (outer-inner) * split(splitter,seed)

            return border       
        else:
            #TODO check if this can be changed for no infinite expand
            # Here large expansion is not that severe
            return c*INFINITY
    finally:
        free(best_y_box_in)
        free(best_y_box_out)


@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double gini_impurity(int* y_in,int* y_out, int[::1] cw) nogil:
    #only works for 2 classes!
    cdef int n_classes = 2
    cdef int len_in,len_out,tot_len,i
    cdef double div1,div2,o1,o2,pow1,pow2,prod,gini_score,weight_tot
    cdef double imp_in = 1, imp_out = 1
    cdef double weight_in = 0,weight_out =  0

    len_in = y_in[0] + y_in[1]
    len_out = y_out[0] + y_out[1]
    tot_len = len_in + len_out

    o2 = len_in
    if len_in > 0:
        for i in range(n_classes):
            o1 = y_in[i]
            div1 = o1 / o2
            pow1 = cpow(div1,2)
            imp_in -= pow1
    else:
        imp_in = 0

    o2 = len_out
    if len_out > 0:
        for i in range(n_classes):
            o1 = y_out[i]
            div2 = o1 / o2
            pow2 = cpow(div2,2)
            imp_out -= pow2
    else:
        imp_out = 0


    for i in range(n_classes):
        prod = y_in[i] * cw[i]
        weight_in += prod / tot_len
        prod = y_out[i] * cw[i]
        weight_out += prod / tot_len

    weight_tot = weight_in + weight_out
    if weight_tot == 0:
        gini_score = 0
        return gini_score

    o1 = imp_in * weight_in
    o2 = imp_out * weight_out
    div1 = o1 / weight_tot
    div2 = o2 / weight_tot

    gini_score =  div1 + div2
    return gini_score

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef double find_closest_rare_overall(double [::1,:] x,int [::1] y,int pt_idx, int f,int c,int splitter,int seed,double eps,double* x_max,double* x_min) nogil:
    cdef int i 
    cdef int p = 0
    cdef double inner = c*INFINITY
    cdef double border
    cdef double val
    cdef double outer
        
    for i in range(x.shape[0]):
        if (y[i] == 1) and (i != pt_idx):
            if c*x[i,f] > c*x[pt_idx,f]:
                if c*x[i,f] < c*inner:
                    inner = x[i,f]

    if inner == c*INFINITY:
        #When there is no rare point at all in this direction dont expand too much -> otherwise load to many leaves
        if c==1:
            outer = x_max[f]
        else:
            outer = x_min[f]
        val = (outer - x[pt_idx,f]) * split(3,seed) #min TODO or random? 
        if val == 0.:
            val = eps
        border = x[pt_idx,f] + c * fabs(val)
        return border
    else:
        #val = bound * 0.0001
        val = (inner - x[pt_idx,f]) * split(splitter,seed)
        if val == 0.:
            val = eps
        border = inner + c * fabs(val)
        
        return border

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef double find_furthest_rare_overall(double [::1,:] x,int [::1] y,double threshold, int f,int c,int splitter, int seed,double eps,double* x_max, double* x_min) nogil:
    cdef int i 
    cdef int p = 0
    cdef double inner = -1 * c*INFINITY
    cdef double border
    cdef double val
    cdef double outer
        
    for i in range(x.shape[0]):
        if (y[i] == 1) :
            if c*x[i,f] > c*threshold:
                if c*x[i,f] > c*inner:
                    inner = x[i,f]

    if inner == -1 * c*INFINITY:
        if c==1:
            outer = x_max[f]
        else:
            outer = x_min[f]
        val = (outer - threshold) * split(3,seed) #min? TODO or random?
        if val == 0.:
            val = eps
        border = threshold + c * fabs(val)
        return border
    else:
        #val = bound * 0.0001
        val = (inner - threshold) * split(splitter,seed)
        if val == 0.:
            val = eps
        border = inner + c * fabs(val)
    return border

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef int filter_all(double[:,::1] bbox,  double[::1, :] x, int[::1] pts_idx) nogil:
    cdef int i,idx,count
    cdef int ncount = 0
    
    for i in range(pts_idx.shape[0]):
        idx,count = 0,0
        while (idx < bbox.shape[0]) and (count == idx):
        #for idx in range(minX.shape[0]):
            if (x[i,idx] >= bbox[idx,0]) and (x[i,idx] <= bbox[idx,1]):
                count += 1
            idx += 1
        if count == idx:
            pts_idx[i] = 1
            ncount += 1
        else:
            pts_idx[i] = 0
    return ncount

# Filter function
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef int filter_all_red(double[:,::1] bbox,  double[::1, :] x, int[::1] pts_idx,int* dims,int L_dims) nogil:
    cdef int i,idx,count,fidx
    cdef double minX,maxX
    cdef int ncount = 0

    for i in range(pts_idx.shape[0]):
        count,idx = 0,0
        while (idx < L_dims) and (count == idx):
            fidx = dims[idx]
            minX = bbox[fidx,0]
            maxX = bbox[fidx,1]
            if (x[i,fidx] >= minX) and (x[i,fidx] <= maxX):
                count += 1
            idx += 1
        if count == idx:
            pts_idx[i] = 1
            ncount += 1
        else:
            pts_idx[i] = 0
    return ncount

##############
#filter_all function with Python compatibility
##############
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cpdef int filter_py(double[:,::1] bbox,  double[::1, :] x, int[::1] pts_idx):
    cdef int i,idx,count
    cdef int ncount = 0
    
    for i in range(pts_idx.shape[0]):
        idx,count = 0,0
        while (idx < bbox.shape[0]) and (count == idx):
        #for idx in range(minX.shape[0]):
            if (x[i,idx] >= bbox[idx,0]) and (x[i,idx] <= bbox[idx,1]):
                count += 1
            idx += 1
        if count == idx:
            pts_idx[i] = 1
            ncount += 1
        else:
            pts_idx[i] = 0
    return ncount

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
@cython.nonecheck(False)
cdef int check_complete_duplicates(int[::1] y,int pt_idx,int[::1] cw, int[::1] pts_idx,int *y_box_in,int *y_box_out) nogil:
    cdef double score_all,score_single

    cdef int* y_box_in_single = <int*> malloc(2 * sizeof(int))
    cdef int* y_box_out_single = <int*> malloc(2 * sizeof(int))
    
    try:
        y_box_in_single[1] = 1
        y_box_in_single[0] = 0

        y_box_out_single[1] = y_box_out[1]
        y_box_out_single[0] = y_box_out[0]


        for i in range(y.shape[0]):
            if pts_idx[i] == 1:
                if i != pt_idx:
                    if y[i] == 1:
                        y_box_in[1] += 1
                        y_box_out[1] -= 1
                    else:
                        y_box_in[0] += 1
                        y_box_out[0] -= 1

        score_all = gini_impurity(y_box_in,y_box_out,cw)
        score_single = gini_impurity(y_box_in_single,y_box_out_single,cw)
        if score_single < score_all:
            return 1
        else: 
            return 0
    finally:
        free(y_box_in_single)
        free(y_box_out_single)


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