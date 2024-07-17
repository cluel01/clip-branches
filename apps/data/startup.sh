#!/bin/bash

# sed -i "s/%COUNTRY_CODE%/$COUNTRY_CODE/g" .env
# sed -i "s/%COUNTRY_CODE%/${COUNTRY_CODE,,}/g" app/config.py

# sed -i "s/%EPSG%/$EPSG/g" .env

# country_tif_path=$(echo $COUNTRY_TIF_PATH | sed 's/\//\\\//g') # escape slashes
# sed -i "s/%COUNTRY_TIF_PATH%/$country_tif_path/g" .env

uvicorn app.main:app --workers $NUM_WORKERS --host 0.0.0.0 --port $PORT