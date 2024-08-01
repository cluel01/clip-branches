DATA_NUM_WORKERS=1

docker run -d --rm  -p 5000:80    -v ./assets/search/config:/usr/src/app/assets/config:ro \
  -v ./assets/search/data:/usr/src/app/assets/data:ro \
   -v ./data/indexes:/usr/src/app/indexes --name search search

docker run --rm -d -p 5001:8000 -v ./assets/data/:/opt:ro -v ./assets/data/.env:/usr/src/app/.env:ro -v  ./data/datasets:/mnt:ro --name data data

docker run --rm -d -p 8888:3000 -v ./assets/web/.env:/usr/src/app/.env:ro --name web web