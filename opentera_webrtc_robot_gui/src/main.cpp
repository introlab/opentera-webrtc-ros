#include "MainWindow.h"

#include <QApplication>
#include <QImageReader>
#include <QImage>
#include <QDebug>
#include <ros/ros.h>
#include <signal.h>
#include <QThread>
#include <QFile>

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

    //Stylesheet
    QFile file(":/stylesheet.qss");
    file.open(QFile::ReadOnly);
    QString stylesheet = QLatin1String(file.readAll());
    app.setStyleSheet(stylesheet);


    MainWindow w;
    w.show();


    //Load test image from QRC
    QImage testImage(":/Test_640x480.png");

    qDebug() << testImage;
    w.setImage(testImage);

    //Will start ROS loop in background
    spinner.start();

    //will run app event loop (infinite)
    //qDebug() << "MainThread " << QThread::currentThread();
    app.exec();

    //Stop ROS loop
    spinner.stop();
}
