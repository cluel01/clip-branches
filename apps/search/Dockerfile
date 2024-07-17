# https://hub.docker.com/_/python
FROM python:3.10

WORKDIR /usr/src/app

# set environment variables for python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install requirements
COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache -r requirements.txt


#Install decisionbranches
COPY ./decisionbranches/. ./decisionbranches
RUN pip3 install --no-cache-dir decisionbranches/.


#Copy clip
COPY ./clip ./clip

#Install boxnet
COPY ./boxnet/. ./boxnet
RUN pip3 install --no-cache-dir boxnet/.

ENV SEARCH_STORAGE_PATH /tmp/search_results

# copy project
COPY ./app .

EXPOSE 80

# For replication we use the kubernetes way as documented:
# https://fastapi.tiangolo.com/deployment/concepts/#examples-of-replication-tools-and-strategies
CMD ["uvicorn", "main:app", "--host=0.0.0.0" , "--port", "80"]
