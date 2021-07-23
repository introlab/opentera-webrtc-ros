#include "MainWindow.h"

#include <QApplication>
#include <QImageReader>
#include <QImage>
#include <QDebug>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
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

    return a.exec();



}
