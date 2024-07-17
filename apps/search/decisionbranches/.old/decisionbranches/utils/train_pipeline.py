import numpy as np
from numpy.random import default_rng
import warnings
from time import time,strftime
import os
import logging
import sys
import json

import tqdm.notebook as tq

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import UndefinedMetricWarning

from ..models.boxSearch.boxClassifier import BoxClassifier
from ..models.boxSearch.ensemble import Ensemble

def run_experiment(datasets,model,model_params,experiment_params,param_grid,seed,log_file=None,progress_bar=True):
    np.random.seed(seed)
    
    logger_name = "run_"+model+"_"+strftime("%Y%m%d-%H%M%S")
    if log_file is None:
        log_file= logger_name+".log"
    
    if os.path.isfile(log_file):
        return f"Log file {log_file} is already existing!"
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("logger")
    
    results = []
       
    if datasets == "all":
        #datasets = {"iris":{"labels":"all"},"covtype":{"labels":"all"},"satimage":{"labels":"all"},
        #            "senseit":{"labels":"all"},"mnist":{"labels":"all"},"artificial":{"labels":[1]},"letter":{"labels":"all"}}
        datasets = {"iris":{"labels":"all"},"covtype":{"labels":"all"},"satimage":{"labels":"all"},
            "senseit":{"labels":"all"},"mnist":{"labels":"all"},"letter":{"labels":"all"}}
    
    
    #################Experiment########################
    check_param = [k for k,v in experiment_params.items() if type(v) == list]
    
    if (len(check_param) > 1):
        logger.error("Only one parameter can be tested at once!")
        return 
    elif len(check_param) == 0:
        logger.info("No experimental parameter chosen")
        eparams = [None]
    else:
        check_param = check_param[0]
        logger.info(f"Parameter: {check_param}")
        eparams = experiment_params[check_param]
    
    
    
    ############# Model #################
    logger.info(f"Model: {model}")
    if len(model_params) == 0:
        logger.info("No model parameter given")
        mparams = [None]
    else:
        check_param_model = [k for k,v in model_params.items() if type(v) == list]
        if (len(check_param_model) > 1) or ((len(check_param) == 1) and (eparams[0] != None)):
            logger.error("Only one parameter can be tested at once!")
            return 
        elif len(check_param_model) == 0:
            logger.info("No model parameter chosen")
            mparams = [None]
        else:
            check_param = check_param_model[0]
            logger.info(f"Parameter: {check_param}")
            mparams = model_params[check_param]
    
    if progress_bar:
        if mparams[0] is not None:
            pbar = tq.tqdm(total=len(mparams)*len(datasets), position=0,leave=True)
        elif eparams[0] is not None:
            pbar = tq.tqdm(total=len(eparams)*len(datasets), position=0,leave=True)
        else:
            pbar = tq.tqdm(total=len(datasets), position=0,leave=True)

    for e in eparams:
        for v in mparams:
            for dset,dparams in datasets.items():
                logger.info(f"Dataset: {dset}")
                if v is not None:
                    model_params[check_param] = v
                elif e is not None:
                    experiment_params[check_param] = e

                res = train_model(dset,model,model_params,dparams,experiment_params,param_grid,seed,logger_name)
                if progress_bar:
                    pbar.update(1)
                    logger.info(f"Run {pbar.n}/{pbar.total}")
                if res is None:
                    logger.info(f"Skip dataset {dset}!")
                    continue
                results.extend(res)
                
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    return results


def train_model(dataset,model_type,model_params,dataset_params,experiment_params,param_grid,seed,logger_name):
    logger = logging.getLogger(logger_name)
    X,y = get_dataset(dataset,seed,dataset_params)
    
    labels = dataset_params["labels"]
    if labels == "all":
        labels = np.unique(y).tolist()
    
    results = []
    
    for l in labels:
        for seed in range(experiment_params["n_seeds"]):
            data = make_splits(X=X,y=y,label=l,seed=seed,experiment_params=experiment_params,logger_name=logger_name)
            if data is not None:
                X_train,X_val,X_test,y_train,y_val,y_test = data
            else:
                logger.error("Error in data splitting!")
                return

            if (model_type == "dbranch") or (model_type == "ensemble"):
                model_params["tot_feat"] = X.shape[1]
                model_params["n_rare"] = experiment_params["n_rare"]

            start_validate = time()
            model_config = grid_search(X_train,X_val,y_train,y_val,model_type,seed,experiment_params,param_grid,model_params)
            end_validate = time()

            model = get_model(model_type,model_config,seed,model_params)

            start = time()
            model.fit(X_train,y_train)
            end = time()
            
            pred_train =model.predict(X_train)
            start_pred = time()
            pred = model.predict(X_test)
            end_pred = time()

            training_time = end-start
            validation_time = end_validate-start_validate
            pred_time = end_pred-start_pred
            train_score = get_scores(pred_train,y_train,prefix="train_")
            test_score = get_scores(pred,y_test,prefix="test_")

            results.append({"model": model_type,"dataset":dataset,"label":l,**model_params,**dataset_params,**experiment_params,
                            **train_score,**test_score,"training_time":training_time,"validation_time":validation_time,"prediction_time":pred_time,
                            "model_cfg": str(model_config),"seed":seed})
            logger.info(f"Result of run: {json.dumps(results[-1])}")
    return results


#### Datasets

def get_dataset(dataset,seed,data_params):
    dparams = data_params.copy()
    dparams.pop("labels")
    if dataset == "iris":
        return get_iris()
    if dataset == "covtype":
        return get_covtype(**dparams)
    if dataset == "artificial":
        return get_artificial(seed=seed,**dparams)
    if dataset == "mnist":
        return get_mnist()
    if dataset == "letter":
        return get_letter()
    if dataset == "senseit":
        return get_senseit()
    if dataset == "satimage":
        return get_satimage()
    print(f"Error not existing dataset: {dataset}")

def get_iris():
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    return X,y

def get_covtype(data_home="~/work/data/external"):
    X,y = fetch_covtype(return_X_y=True,data_home=data_home)
    return X,y

def get_artificial(n_features=20,n_samples=100_000,cfg=None,seed=42):
    if cfg is None:
        cfg = {"class_sep":2,"weights":[0.999,0.001],"n_informative":10,"n_redundant":0,"flip_y":0,"n_clusters_per_class":1}
    X, y = make_classification(n_classes=2,n_samples=n_samples,n_features=n_features,random_state=seed,**cfg)
    return X,y

def get_mnist():  
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    return X,y

def get_letter():
    X, y = fetch_openml('letter', version=1, return_X_y=True, as_frame=False)
    #Remove duplicates
    #idx = np.unique(X,axis=0,return_index=True)[1]
    #X = X[idx]
    #y = y[idx]
    return X,y

def get_senseit():
    X, y = fetch_openml('SensIT-Vehicle-Combined', version=1, return_X_y=True, as_frame=False)

    X = X.toarray()
    return X,y

def get_satimage():
    X, y = fetch_openml('satimage', version=1, return_X_y=True, as_frame=False)
    return X,y


def get_model(model,cfg,seed,model_params):
    ####cfg -> tuned parameter | model_params -> fixed
    if model == "dtree":
        return get_dtree(cfg=cfg,seed=seed,model_params=model_params)
    if model == "rf":
        return get_rf(cfg=cfg,seed=seed,model_params=model_params)
    if model == "extra":
        return get_extra(cfg=cfg,seed=seed,model_params=model_params)
    if model == "dbranch":
        return get_decisionbranch(cfg=cfg,seed=seed,model_params=model_params)
    if model == "ensemble":
        return get_ensemble(cfg=cfg,seed=seed,model_params=model_params)
    print(f"Error not existing model: {model}")

def get_dtree(cfg,seed,model_params):
    params = {**model_params,**cfg} #merge two and override duplicates by the optimized
    if "max_features" in params:
        if params["max_features"] == "all":
            params["max_features"] = None
    dtree = DecisionTreeClassifier(**params,random_state=seed)
    return dtree

def get_rf(cfg,seed,model_params):
    params = {**model_params,**cfg} #merge two and override duplicates by the optimized
    if "max_features" in params:
        if params["max_features"] == "all":
            params["max_features"] = None
    rf = RandomForestClassifier(**params,random_state=seed)
    return rf

def get_extra(cfg,seed,model_params):
    params = {**model_params,**cfg} #merge two and override duplicates by the optimized
    if "max_features" in params:
        if params["max_features"] == "all":
            params["max_features"] = None
    et = ExtraTreesClassifier(**params,random_state=seed)
    return et

def get_decisionbranch(cfg,seed,model_params):
    mp = model_params.copy()
    c = cfg.copy()
    
    ni = mp.pop("n_ind")
    me = c.pop("max_features")
    
    nind = int(ni * mp["tot_feat"])
    c["max_evals"] = me
    if "stop_infinite" in mp:
        c["stop_infinite"] = mp.pop("stop_infinite")      
    
    n_rare = mp.pop("n_rare")
    if "min_pts" in c:
        if c["min_pts"] == "auto":
            c["min_pts"] = n_rare
    
    se = BoxClassifier(cfg=c,n_ind=nind,**mp,seed=seed)
    return se

def get_ensemble(cfg,seed,model_params):
    mp = model_params.copy()
    c = cfg.copy()
    
    if "bootstrap" in c:
        bt = c.pop("bootstrap")
        mp["bootstrap"] =bt
    
    if "n_estimators" in c:
        ne = c.pop("n_estimators")
        mp["n_estimators"] = ne
    
    ni = mp.pop("n_ind")
    me = c.pop("max_features")
    nind = int(ni * mp["tot_feat"])
    
    n_rare = mp.pop("n_rare")
    if "min_pts" in c:
        if c["min_pts"] == "auto":
                c["min_pts"] = n_rare
    
    c["max_evals"] = me
    if "stop_infinite" in mp:
        c["stop_infinite"] = mp.pop("stop_infinite")

    ens = Ensemble(cfg=c,n_ind=nind,**mp,seed=seed)
    return ens

def get_scores(y_pred,y_true,prefix=""):
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    score = {}
    score[prefix+"accuracy"] = accuracy_score(y_true,y_pred)
    score[prefix+"precision"] = precision_score(y_true,y_pred)
    score[prefix+"recall"] = recall_score(y_true,y_pred)
    score[prefix+"f1"] = f1_score(y_true,y_pred)
    score[prefix+"balanced_accuracy"] = balanced_accuracy_score(y_true,y_pred)
    return score


def make_splits(X,y,label,seed,experiment_params,logger_name):
    rng = default_rng(seed)
    logger = logging.getLogger(logger_name)
    
    n_rare = experiment_params["n_rare"]
    val_set = experiment_params["val_set"]
    if val_set:
        n_rare_val = experiment_params["n_rare_val"]
    scaling = experiment_params["scaling"]
    
    y_bin = np.zeros(len(y),dtype=int)
    y_bin[y==label] = 1
    rare = np.where(y_bin == 1)[0]
    nonrare = np.where(y_bin == 0)[0]
    
    if n_rare >= len(rare):
        logger.error("Not enough points for training!")
        return
    
    rare_train = rare[rng.choice(np.arange(len(rare)),size=n_rare,replace=False)]
    
    class_dist = n_rare / len(rare)
    n_nonrare = int(class_dist * len(nonrare))
    nonrare_train = nonrare[rng.choice(np.arange(len(nonrare)),size=n_nonrare,replace=False)]
    
    rare_test = np.setdiff1d(rare,rare_train)
    nonrare_test = np.setdiff1d(nonrare,nonrare_train)
    
    if (len(rare_test) == 0) or (len(nonrare_test) == 0):
        logging.error("No points remaining!")
        return
    
    if val_set:
        if n_rare_val ==  "train":
            n_rare_val = n_rare
        elif type(n_rare_val) == float:
            n_rare_val = round((np.sum(y_bin)-n_rare)*n_rare_val)
            
        
        if n_rare_val >= len(rare_test):
            logger.error("Not enough points remaining for validation and test!")
            return
        rare_val = rare_test[rng.choice(np.arange(len(rare_test)),size=n_rare_val,replace=False)]
        n_nonrare = int(class_dist * len(nonrare_test))
        nonrare_val = nonrare_test[rng.choice(np.arange(len(nonrare_test)),size=n_nonrare,replace=False)]
        
        rare_test = np.setdiff1d(rare_test,rare_val)
        nonrare_test = np.setdiff1d(nonrare_test,nonrare_val)
        
        train_idx = np.concatenate([rare_train,nonrare_train]) 
        rng.shuffle(train_idx)
        
        val_idx = np.concatenate([rare_val,nonrare_val]) 
        rng.shuffle(val_idx)
        
        test_idx = np.concatenate([rare_test,nonrare_test]) 
        rng.shuffle(test_idx)
        X_train,X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train,y_val, y_test = y_bin[train_idx], y_bin[val_idx], y_bin[test_idx]

        if scaling:
            std_scaler = StandardScaler()
            #minmax_scaler = MinMaxScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_val = std_scaler.transform(X_val)
            X_test = std_scaler.transform(X_test)
        

        return (X_train,X_val,X_test,y_train,y_val,y_test)
        
        
        
    else:
        train_idx = np.concatenate([rare_train,nonrare_train]) 
        rng.shuffle(train_idx)
        test_idx = np.concatenate([rare_test,nonrare_test]) 
        rng.shuffle(test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]
        
        if scaling:
            std_scaler = StandardScaler()
            #minmax_scaler = MinMaxScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)

        return (X_train,None,X_test,y_train,None,y_test)


def grid_search(X_train,X_test,y_train,y_test,model_type,seed,experiment_params,param_grid,model_params):
    val_criterion = experiment_params["val_criterion"]
    val_set = experiment_params["val_set"]
    if val_set == False:
        cv_folds = experiment_params["cv_folds"]
        skf = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=seed)

    grid = ParameterGrid(param_grid)
    best_score = -np.inf
    best_params = None
    for params in grid:
        if val_set:
            model = get_model(model=model_type,cfg=params,seed=seed,model_params=model_params)
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            scores = get_scores(pred,y_test)
            score = scores[val_criterion]
        else:
            all_scores = []
            for train_idx,test_idx in skf.split(X_train,y_train):
                model = get_model(model=model_type,cfg=params,seed=seed,model_params=model_params)
                xtrain,xtest = X_train[train_idx], X_train[test_idx]
                ytrain,ytest = y_train[train_idx], y_train[test_idx]
                model.fit(xtrain,ytrain)
                pred = model.predict(xtest)
                scores = get_scores(pred,ytest)
                score = scores[val_criterion]
                all_scores.append(score)
            score = np.mean(all_scores)
        
        if score > best_score:
            best_params = params
            best_score = score
            
    return best_params
        
        
        
            