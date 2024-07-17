"""
Main module for the FastAPI search service.
"""

from init import initialize_searcher
from utils import quantize_features,store_results
from config import Settings

import os
import pathlib
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import time
import numpy as np
from typing import List
from pydantic import BaseModel



class SearchIndices(BaseModel):
    idxs_rare: List[int] = []
    idxs_nonrare: List[int] = []

start_time = time.time()
print("service starting!")

# This environment variable has to be set to avoid errors. It is a workaround but we did not experience any unwanted behavior.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

STORAGE_PATH = os.environ['SEARCH_STORAGE_PATH']
os.makedirs(STORAGE_PATH, exist_ok=True)

app = FastAPI()

# # We added wildcard CORS middleware for development. For production use this should be refined to only allow valid requests.
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

#Initialization of the search service
cfg = Settings.Config.cfg
searchers = initialize_searcher(cfg)

# No empty response
@app.get("/")
async def redirect():
    """
    Empty requests redirect to the docs page.
    """
    return RedirectResponse(url="/docs")


@app.get("/ping")
async def ping():
    """Healthcheck endpoint"""
    return "pong!"

@app.get("/get_available_searchers")
async def get_available_searchers():
    s_dict = {}
    cfg = Settings.Config.cfg
    for dataset in cfg.keys():
        s_dataset = []
        for s in cfg[dataset]["searcher"].keys():
            searcher_display_name = cfg[dataset]["searcher"][s]["display_name"]
            searcher_id = dataset + "_" + s
            s_dataset.append([searcher_display_name,searcher_id])
        s_dict[dataset] = {"dataset_cfg":{},"searchers":s_dataset,"dataset_display_name":cfg[dataset]["display_name"],
                        "max_nonrare_samples":None}
    return JSONResponse(s_dict)



@app.post("/search/{dataset}/{searcher_name}/")
async def search(dataset: str, searcher_name: str, search_indices: SearchIndices,n_nonrare_samples: int,negative_weight:int,background_tasks: BackgroundTasks,nresults=None,
                 first_batch_size=30):
    sql_statements = []

    start_time = time.time()
    print("started search!")
    if searcher_name.startswith(dataset):
        db = searcher_name[len(dataset)+1:]
    else:
        db = searcher_name

    np.random.seed(int(searchers[dataset]["seed"]))

    idxs_rare = np.array(search_indices.idxs_rare).astype(int)
    idxs_nonrare = np.array(search_indices.idxs_nonrare).astype(int)

    assert negative_weight > 0
    if negative_weight > 1:
        print(f"INFO: Weighted selected negative elements by factor {negative_weight}!")
        idxs_nonrare = idxs_nonrare.repeat(negative_weight)
    
    db_model = searchers[dataset]["searcher"][db]
    features = searchers[dataset]["features"]
    treeset = searchers[dataset]["kdt_search"]
    nonrare_feat =  searchers[dataset]["non_rare_feat"]

    assert (len(idxs_rare) > 0)
    if not db.startswith("nn") and n_nonrare_samples <= 0:
        assert len(idxs_nonrare) > 0 

    idxs_rare.sort()
    idxs_nonrare.sort()

    start_load = time.time()
    idxs = np.hstack([idxs_rare,idxs_nonrare])
    order = idxs.argsort()
    idxs = idxs[order]
    X_train = features[idxs]
    
    if X_train.dtype != np.dtype(db_model.dtype):
        X_train = X_train.astype(db_model.dtype)

    start_nonrare_sampling = time.time()
    if (nonrare_feat is not None) and (n_nonrare_samples > 0):
        print("INFO: Included extra non rare elements")
        if nonrare_feat.dtype != np.dtype(db_model.dtype):
            print("INFO: dtype of non rare features needed to be transformed")
            nonrare_feat = nonrare_feat.astype(db_model.dtype)

        nonrare_idxs = np.in1d(searchers[dataset]["non_rare_idxs"],idxs_rare)
        nonrare_feat = nonrare_feat[~nonrare_idxs]

        if len(nonrare_feat) > n_nonrare_samples:
            subset_idxs = np.random.choice(np.arange(len(nonrare_feat)),size=n_nonrare_samples,replace=False)
            nonrare_subset = nonrare_feat[subset_idxs]
        else:
            nonrare_subset = nonrare_feat 

        X_train = np.vstack([X_train,nonrare_subset])
        
        add_non = np.arange(len(order),len(order)+len(nonrare_subset))
        order = np.hstack([order,add_non])
    end_nonrare_sampling = time.time()

    print("Num positives: ",len(idxs_rare))
    print("Num negatives: ",len(idxs_nonrare))
    print("Num negatives (total): ",len(X_train)-len(idxs_rare))
    print("Num samples: ",X_train.shape)

    y_train = np.zeros(len(X_train))
    y_train[:len(idxs_rare)] = 1
    y_train = y_train[order]
    end_load = time.time()

    start_model = time.time()
    if db.startswith("db") or db.startswith("ens"):
        if db.startswith("db"):
            inds,sql_statements = db_model.search(X_train, y_train,features,treeset)
        else:
            inds = db_model.search(X_train, y_train,features,treeset)
    elif db.startswith("nn"):
        inds = db_model.search(X_train, y_train,treeset)
    elif db.startswith("boxnet"):
        inds,sql_statements = db_model.search(X_train, y_train,treeset)
    else:
        inds = db_model.search(X_train, y_train,features)
    end_model = time.time()
    
    search_id = str(start_time).replace(".","")
    background_tasks.add_task(store_results, inds,search_id,STORAGE_PATH)

    total_results = len(inds)   
    if nresults is not None:
        nresults = int(nresults)
        inds = inds[:nresults]

    

    # inds_sub = inds[:first_batch_size]
    # start_save = time.time()
    # if len(inds) <= first_batch_size:
    #     background_tasks.add_task(store_results, inds[first_batch_size:],search_id)
    # else:
    #     search_id = None
    # end_save = time.time()

    end_time = time.time()
    return JSONResponse({"results":inds.tolist(),"time":end_time-start_time,"nresults":total_results,"search_id":search_id,"sql_statements":sql_statements})

@app.get("/search_text/{dataset}/{text}")
async def search_text(dataset: str,text: str,nresults=100):
    start_time = time.time()

    nresults = int(nresults)
    cfg = searchers[dataset]["cfg"]["kdt_textsearch"]
    
    kdtree= searchers[dataset]["kdt_text"]
    if cfg["type"] == "treeset":
        kdtree = searchers[dataset]["kdt_search"]

    model = searchers[dataset]["text_model"]

    start_encode = time.time()
    model.eval()
    
    query = model.encode_text(text,tokenize=True).detach().cpu().numpy().reshape(-1)#.astype(kdtree.dtype)
    
    if cfg["type"] != "faiss":
        if kdtree.dtype == "uint8":
            query = quantize_features(query)
        query = query.astype(kdtree.dtype)
    else:
        if searchers[dataset]["features"].dtype == "uint8":
            query = quantize_features(query)
        query = query.astype(np.float32)

    if searchers[dataset]["feat_sub"] is not None:
        query = query[cfg["feat_sub"]]
    end_encode = time.time()

    if "stop_leaves" in cfg.keys():
        stop_leaves = cfg["stop_leaves"]
    else:
        stop_leaves = None

    start_search = time.time()
    if cfg["type"] == "treeset":
        inds,counts,_,loaded_leaves = kdtree.multi_query_nn_cy(query,k=nresults,stop_leaves=stop_leaves)
        inds = inds[:nresults]
        print("counts: ",counts[:nresults])
    elif cfg["type"] == "faiss":
        _,inds = kdtree.search(query.reshape(1,-1), k=nresults)
        inds = inds[0]
        loaded_leaves = None
    else:
        inds,_,_,loaded_leaves = kdtree.query_point_cy(query,k=nresults,stop_leaves=stop_leaves)

    end_search = time.time()

    end_time = time.time()

    print(f"******************")
    print("inds: ",inds)
    print("encode text: ",text)
    print("query shape: ",query.shape)
    print(f"Encoding took {end_encode-start_encode: .3f} s")
    print(f"Search took {end_search-start_search: .3f} s")
    print(f"Complete query took {end_time-start_time: .3f} s")
    print(f"Loaded leaves: {loaded_leaves}")
    print(f"******************")
    return JSONResponse({"results":inds.tolist(),"time":end_time-start_time})

@app.get("/get_search_results/{search_id}")
async def get_search_results(search_id: str):
    filename = os.path.join(STORAGE_PATH,search_id+".npy")
    if os.path.isfile(filename):
        windows = np.load(filename)
        os.remove(filename)
        return JSONResponse({"results":windows.tolist()})
    else:
        return JSONResponse({"results":None})
    


print("server took ", time.time()-start_time, "s to start")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)


