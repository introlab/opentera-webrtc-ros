#!/bin/bash          

echo "Starting opentera signaling server from :" $1 
port=$2
password=$3
static_folder=$4

echo "port :" $port
echo "password:" $password
echo "static_folder:" $static_folder
echo "Client interface will be available at http://localhost:$port/index.html?pwd=$password"

(cd $1; python3 -m pip install -r requirements.txt;python3 signaling_server.py --port $port --password $password --static_folder $static_folder)
