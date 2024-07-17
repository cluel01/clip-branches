# pull official base image
FROM python:3.10

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# copy requirements file
COPY ./requirements.txt /usr/src/app/requirements.txt

# install requirements in separate step for caching
RUN set -e && apt-get update && apt-get install -y python3-pip

# install dependencies
RUN set -u \
    # && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --upgrade -r /usr/src/app/requirements.txt

# copy project
COPY . /usr/src/app/

EXPOSE 8000

ENV PORT 8000
ENV NUM_WORKERS 1

RUN chmod u+x startup.sh
CMD ["/bin/sh", "-c", "./startup.sh"] 