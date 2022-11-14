#!/bin/bash

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`

mkdir -p $SCRIPT_PATH/../models

# Download model
cd $SCRIPT_PATH/../models
OUT=$(wget -nc https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt 2>&1) 
RETVAL=$?
if [ $RETVAL != 0 ]; then
    echo $OUT 1>&2
    exit $RETVAL
fi

OUT=$(wget -nc https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel 2>&1)
RETVAL=$?
if [ $RETVAL != 0 ]; then
    echo $OUT 1>&2
    exit $RETVAL
fi
