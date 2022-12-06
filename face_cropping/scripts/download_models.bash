#!/bin/bash

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`

mkdir -p $SCRIPT_PATH/../models

# Download model
cd $SCRIPT_PATH/../models
OUT=$(wget -nc https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml 2>&1) 
RETVAL=$?
if [ $RETVAL != 0 ]; then
    echo $OUT 1>&2
    exit $RETVAL
fi

OUT=$(wget -nc https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface_improved.xml 2>&1)
RETVAL=$?
if [ $RETVAL != 0 ]; then
    echo $OUT 1>&2
    exit $RETVAL
fi
