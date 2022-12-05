#include "MainWindow.h"

#include <QApplication>
#include <QImageReader>
#include <QImage>
#include <QDebug>
#include <ros/ros.h>
#include <ros/package.h>
#include <signal.h>
#include <QThread>
#include <QFile>

// SIGINT handler, will quit Qt event loop
void termination_handler(int signum)
{
    QApplication::quit();
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "opentera_webrtc_robot_gui_node");
    ros::NodeHandle nh("~");
    bool fullScreen = false;
    nh.getParam("fullScreen", fullScreen);


    std::string packageName, jsonFilePath;
    nh.param<std::string>("name", packageName, "opentera_webrtc_robot_gui");
    nh.param<std::string>(
        "device_properties_path",
        jsonFilePath,
        ros::package::getPath(packageName) + "/src/resources/DeviceProperties.json");

    ros::AsyncSpinner spinner(1);

    /* Set up the structure to specify the action */
    struct sigaction action;
    action.sa_handler = termination_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);

    // Hides an internal error that comes from resizing an QGLWidget, which doesn't affect our use
    qputenv("QT_LOGGING_RULES", QByteArray("*.debug=false;qt.qpa.xcb=false"));

    QApplication app(argc, argv);
    // Stylesheet
    QFile file(":/stylesheet.qss");
    file.open(QFile::ReadOnly);
    QString stylesheet = QLatin1String(file.readAll());
    app.setStyleSheet(stylesheet);


    MainWindow w(QString::fromStdString(jsonFilePath));
    if (fullScreen)
    {
        w.showFullScreen();
    }
    else
    {
        w.show();
    }

    // Load test image from QRC
    QImage testImage(":/Test_640x480.png");

    qDebug() << testImage;
    w.setImage(testImage);

    // Will start ROS loop in background
    spinner.start();

    // will run app event loop (infinite)
    // qDebug() << "MainThread " << QThread::currentThread();
    app.exec();

    // Stop ROS loop
    spinner.stop();

    // Fixes a bug where the program would stay open because of the camera window
    w.closeCameraWindow();
}
