import time as t
import numpy as np
from boxnet.models.boxnetclassifier import BoxNetBranchesClassifier
from decisionbranches.utils.helpers import generate_fidxs


class BoxNetSearchEngine:
    def __init__(self,totfeat,indices,nboxes,scaling,seed,cfg={},dtype="float32"):
        np.random.seed(seed)

        #Box classifier parameters
        self.totfeat = totfeat
        self.nboxes = nboxes
        self.dtype = dtype

        
        if nboxes < len(indices):
            np.random.shuffle(indices)
            indices = indices[:nboxes]
        self.indices = indices

        self.seed = seed
        self.scaling = scaling
        self.model = BoxNetBranchesClassifier(D=totfeat,nboxes=nboxes,feature_subsets=self.indices,
                                              random_state=seed,**cfg)

    def search(self,X_train,y_train,treeset):
        if self.scaling:
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            X_train = (X_train - mean) / std

        self.model.fit(X_train,y_train)

        mins,maxs,fidxs = self.model.get_boxes()

        #Filter out empty boxes
        #TODO make this more clean
        box_mask = self.model.model.block.box_mask.detach().cpu().numpy()

        if box_mask.sum() == 0:
            print("WARNING: No boxes were found!")
            box_mask[0] = True

        mins = mins[box_mask]
        maxs = maxs[box_mask]
        fidxs = fidxs[box_mask]

        print(f"Number of boxes: {len(mins)}")

        #Reverse scaling
        if self.scaling:
            for i in range(len(mins)):
                std_subset = std[fidxs[i]]
                mean_subset = mean[fidxs[i]]
                mins[i] = (mins[i] * std_subset) + mean_subset
                maxs[i] = (maxs[i] * std_subset) + mean_subset

        sql_statements = self._get_sql_query(mins,maxs,fidxs)

        start = t.time()
        #### Query boxes #########
        inds,counts,time,loaded_leaves = treeset.multi_query_ranked_cy(mins,maxs,fidxs.tolist())


        end = t.time()
        print(f"KDtree search took: {end-start:.3f}")

        return inds,sql_statements

    @staticmethod
    def _get_sql_query(mins,maxs,fidxs):
        sql_statements = []

        for box_idx in range(len(mins)):
            conditions = []
            for dim_idx in range(len(fidxs[box_idx])):
                dim = fidxs[box_idx][dim_idx]
                min_val = mins[box_idx][dim_idx]
                max_val = maxs[box_idx][dim_idx]
                condition = f"(t.{dim} BETWEEN {min_val} AND {max_val})"
                conditions.append(condition)
            where_clause = " AND ".join(conditions)
            sql_statement = f"SELECT * FROM table as t WHERE {where_clause};"
            sql_statements.append(sql_statement)

        return sql_statements