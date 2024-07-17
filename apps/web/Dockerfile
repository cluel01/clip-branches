# Use an official Node runtime as a parent image
FROM node:14

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY package.json .

# Install any needed packages specified in package.json
RUN npm install

COPY . .

# Make port 3000 available to the world outside this container
EXPOSE 3000

ENV ENDPOINT_SEARCHSERVICE "search"
ENV ENDPOINT_DATA "data"
ENV DEMO "true"

RUN chmod u+x startup.sh
CMD ["/bin/sh", "-c", "./startup.sh"] 

