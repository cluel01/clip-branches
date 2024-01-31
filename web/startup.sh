#!/bin/bash
sed -i "s/%ENDPOINT_SEARCHSERVICE%/$ENDPOINT_SEARCHSERVICE/g" server.js
sed -i "s/%ENDPOINT_DATA%/$ENDPOINT_DATA/g" server.js
sed -i "s/%ENDPOINT_PROTOCOL%/${ENDPOINT_PROTOCOL,,}/g" server.js
sed -i "s/%DEMO%/${DEMO}/g" server.js

# Run the app when the container launches
npm start