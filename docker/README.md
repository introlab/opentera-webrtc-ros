# Docker Setup for opentera_webrtc_ros

The docker-compose.yaml contains configurations build and launch opentera_webrtc_ros, including GUI support for tools like opentera_gui.

Dockerfile.opentera_ros2 contains build docker build instructions. 

## Instructions

### 1. Navigate to the Docker Directory and Build
Change your current working directory to the folder containing the Docker setup files:
```bash
cd docker
docker compose build
```
### 2. Allow Docker to Access the Host's X Server
source each terminal session or add to bashrc to enable display from inside container.
```
xhost +local:docker
```
### 3. run container 

```
docker compose up -d
```