#!/bin/bash

echo "Starting opentera signaling server"
port=$1
password=$2
static_folder=$3
robot_type=$4
python_exec=$5

echo "port :" $port
echo "password:" $password
echo "static_folder:" $static_folder
echo "robot_type:" $robot_type
echo "Client interface will be available at http://localhost:$port/index.html#/user?pwd=$password&robot=$robot_type"


(exec $python_exec $(which opentera-signaling-server) --port $port --password $password --static_folder $static_folder)
