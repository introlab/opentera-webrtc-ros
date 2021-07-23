#include "MainWindow.h"

#include <QApplication>
#include <QImageReader>
#include <QImage>
#include <QDebug>
#include <ros/ros.h>
#include <signal.h>
#include <QThread>


// SIGINT handler, will quit Qt event loop
void termination_handler(int signum)
{
	QApplication::quit();
}

int main(int argc, char *argv[])
{

    ros::init(argc, argv, "opentera_webrtc_robot_gui_node");
	ros::AsyncSpinner spinner(1);

    /* Set up the structure to specify the action */
    struct sigaction action;
	action.sa_handler = termination_handler;
	sigemptyset(&action.sa_mask);
	action.sa_flags = 0;
	sigaction(SIGINT, &action, NULL);
    

    QApplication app(argc, argv);
    MainWindow w;
    w.show();


    //Load test image from QRC
    QImage testImage(":/Text_640x480.png");

    qDebug() << testImage;
    w.setImage(testImage);
    ROSCameraView* view1 = w.addThumbnailView(testImage, "Camera #1");
    ROSCameraView* view2 = w.addThumbnailView(testImage, "Camera #2");
    ROSCameraView* view3 = w.addThumbnailView(testImage, "Camera #3");
    ROSCameraView* view4 = w.addThumbnailView(testImage, "Camera #4");

    //Will start ROS loop in background
    spinner.start();

    //will run app event loop (infinite)
    //qDebug() << "MainThread " << QThread::currentThread();
    app.exec();

    //Stop ROS loop
    spinner.stop();
}
