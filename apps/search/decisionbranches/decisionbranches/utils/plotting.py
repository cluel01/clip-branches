#Method for plotting the boxes in 2D space (only 2d boxes accepted) in jupyter notebook environment 
#Only works for 2D and jupyter environment
#test_idx referring to dataset
from ..cython.functions import filter_py

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.backends.backend_pdf
import torch
import numpy as np
import pandas as pd



#TODO add 3D support
def plot_boxes(se,X_train=None,X_test=None,y_train=None,y_test=None,box_id="all",limit=1000000,filename=None):
    assert (X_train is not None) or (X_test is not None), "Either X_train or X_test required for plotting"
    assert se.n_feat == 2, "Invalid Operation for non 2D feature indices!" 

    train_on,test_on = False, False

    if X_train is not None:
        train_on = True
        mask_train = np.ones(len(X_train),dtype=bool)
        if isinstance(X_train,pd.DataFrame):
            X_train = np.array(X_train,dtype=se.dtype)
        elif isinstance(X_train,torch.Tensor):
            X_train = X_train.numpy().astype(se.dtype)
        elif isinstance(X_train,np.ndarray):
            if X_train.dtype != se.dtype:
                X_train = X_train.astype(se.dtype)
    
    if X_test is not None:
        test_on = True
        mask_test = np.ones(len(X_test),dtype=bool)
        if isinstance(X_test,pd.DataFrame):
            X_test = np.array(X_test,dtype=se.dtype)
        elif isinstance(X_test,torch.Tensor):
            X_test = X_test.numpy().astype(se.dtype)
        elif isinstance(X_test,np.ndarray):
            if X_test.dtype != se.dtype:
                X_test = X_test.astype(se.dtype)

    boxes = np.array([i[0] for i in se.bboxes])
    feat_idxs = np.array([i[1] for i in se.bboxes])

    if box_id != "all":
        if isinstance(box_id,int):
            boxes = [boxes[box_id]]
        else:
            boxes = boxes[box_id]

    if filename is not None:
        if not filename.endswith(".pdf"):
            filename = filename + ".pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename)

    for b,f in zip(boxes,feat_idxs):
        fig, ax = plt.subplots()
        ax.set_facecolor((0,0,1,0.7))
        f1,f2 = f
        x1,x2 = b
        
        if train_on:
            #Train
            ax.scatter(X_train[mask_train,f1], X_train[mask_train,f2], c=y_train[mask_train], cmap=matplotlib.colors.ListedColormap(["b","r"]),
                        edgecolor='black',zorder=1,linewidths=1.5,s=30)
                        #Legend
            leg_circle = mlines.Line2D([], [],color="black", marker='o', linestyle='None',
                                    markersize=6, label='Training')

            #remove included points
            pts_box = np.zeros(np.sum(mask_train),dtype=np.intc)
            _ = filter_py(b,X_train[mask_train][:,f],pts_box)
            pts_box = np.asarray(pts_box,dtype=bool)
            mask = np.where(mask_train==True)[0][pts_box]
            mask_train[mask] = False
        
        if test_on:
            #Test
            ax.scatter(X_test[mask_test, f1], X_test[mask_test, f2], c=y_test[mask_test], cmap=matplotlib.colors.ListedColormap(["b","r"]),
                    edgecolor='black',marker="^",zorder=1,linewidths=1.5,s=30)
            leg_triangle = mlines.Line2D([], [],color="black" ,marker='^', linestyle='None',
                                    markersize=6, label='Test')

            pts_box = np.empty(np.sum(mask_test),dtype=np.intc)
            _ = filter_py(b,X_test[mask_test][:,f],pts_box)
            pts_box = np.asarray(pts_box,dtype=bool)
            mask = np.where(mask_test==True)[0][pts_box]
            mask_test[mask] = False

        #Fix the boundaries in case of infinite rectangles
        xlim = ax.get_xlim()
        ylim =ax.get_ylim()

        x1 = [x if x != np.inf else limit for x in x1]
        x2 = [x if x != np.inf else limit for x in x2]
        x1 = [x if x != -np.inf else -limit for x in x1]
        x2 = [x if x != -np.inf else -limit for x in x2]
        
        up = [x1[1]-x1[0],x2[1]-x2[0]]

        # Create a Rectangle patch
        rect = patches.Rectangle((x1[0],x2[0]),up[0] ,up[1], linewidth=1, edgecolor='black', facecolor=(1,0,0,0.7),alpha=1,zorder=-1)
        
        ax.add_patch(rect)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)          

        if train_on and test_on:            
            plt.legend(loc="upper right",handles=[leg_circle,leg_triangle])
        #elif train_on and not test_on :
        #    plt.legend(loc="upper right",handles=[leg_circle])
        #elif test_on and not train_on:
        #    plt.legend(loc="upper right",handles=[leg_triangle])
        plt.xlabel(str(f1))
        plt.ylabel(str(f2))
        
        plt.show()
        
        if filename is not None:
            pdf.savefig(fig)
    if filename is not None:          
        pdf.close()
