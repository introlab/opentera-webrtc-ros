#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "GraphicsViewToolbar.h"
#include "ROSCameraView.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <QImage>
#include <QSharedPointer>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void setImage(const QImage &image);
    ROSCameraView* addThumbnailView(QImage &image, const QString &label="");

signals:
    void newLocalImage(const QImage& image);

private slots:
    void _onLocalImage(const QImage& image);

private:

    void setupROS();

    void localImageCallback(const sensor_msgs::ImageConstPtr& msg);

    void closeEvent(QCloseEvent *event) override;

    Ui::MainWindow *m_ui;
    //Toolbar
    GraphicsViewToolbar *m_toolbar;
    //Local view
    ROSCameraView *m_cameraView;

    ros::NodeHandle m_nodeHandle;
	ros::Subscriber m_peerImageSubscriber;
    ros::Subscriber m_localImageSubscriber;

};


#endif // MAINWINDOW_H
