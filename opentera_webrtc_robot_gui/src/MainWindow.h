#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "GraphicsViewToolbar.h"
#include "ROSCameraView.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>
#include <QImage>
#include <QSharedPointer>
#include <QMap>

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


signals:
    void newLocalImage(const QImage& image);
    void newPeerImage(const QString &id, const QString &name, const QImage &image);
    void newPeerStatus(const QString &id, const QString &name, int status);

private slots:
    void _onLocalImage(const QImage& image);
    void _onPeerImage(const QString& id, const QString& name, const QImage& image);
    void _onPeerStatus(const QString &id, const QString& name, int status);


private:

    void setupROS();

    void localImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void peerImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr &msg);
    void peerStatusCallback(const opentera_webrtc_ros_msgs::PeerStatusConstPtr &msg);

    void closeEvent(QCloseEvent *event) override;

    Ui::MainWindow *m_ui;
    //Toolbar
    GraphicsViewToolbar *m_toolbar;

    //Remote views
    QMap<QString, ROSCameraView*> m_remoteViews;
    
    //Main View
    ROSCameraView *m_cameraView;

    //ROS
    ros::NodeHandle m_nodeHandle;
	ros::Subscriber m_peerImageSubscriber;
    ros::Subscriber m_localImageSubscriber;
    ros::Subscriber m_peerStatusSubscriber;

};


#endif // MAINWINDOW_H
