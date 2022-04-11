#!/bin/bash

echo "Starting opentera signaling server from :" $1
port=$2
password=$3
static_folder=$4
robot_type=$5
python_exec=$6

echo "port :" $port
echo "password:" $password
echo "static_folder:" $static_folder
echo "robot_type:" $robot_type
echo "Client interface will be available at http://localhost:$port/index.html#/user?pwd=$password&robot=$robot_type"


(cd $1; exec $python_exec opentera-signaling-server --port $port --password $password --static_folder $static_folder)
