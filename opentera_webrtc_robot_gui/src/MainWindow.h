#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "GraphicsViewToolbar.h"
#include "ROSCameraView.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>
#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>
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
    void eventJoinSession(const QString &session_url, 
                            const QString &session_creator_name,
                            const QString &session_uuid,
                            QList<QString> session_participants,
                            QList<QString> session_users,
                            QList<QString> session_devices,
                            const QString &join_msg,
                            const QString &session_parameters,
                            const QString &service_uuid);

    void eventStopSession(const QString &session_uuid, const QString &service_uuid);

    void eventLeaveSession(const QString &session_uuid,
                            const QString &service_uuid,
                            QList<QString> leaving_participants, 
                            QList<QString> leaving_users,
                            QList<QString> leaving_devices);

private slots:
    void _onLocalImage(const QImage& image);
    void _onPeerImage(const QString& id, const QString& name, const QImage& image);
    void _onPeerStatus(const QString &id, const QString& name, int status);

    void _onJoinSessionEvent(const QString &session_url, 
                            const QString &session_creator_name,
                            const QString &session_uuid,
                            QList<QString> session_participants,
                            QList<QString> session_users,
                            QList<QString> session_devices,
                            const QString &join_msg,
                            const QString &session_parameters,
                            const QString &service_uuid);

    void _onStopSessionEvent(const QString &session_uuid, const QString &service_uuid);

    void _onLeaveSessionEvent(const QString &session_uuid,
                            const QString &service_uuid,
                            QList<QString> leaving_participants, 
                            QList<QString> leaving_users,
                            QList<QString> leaving_devices);
   


private:

    void setupROS();

    void localImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void peerImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr &msg);
    void peerStatusCallback(const opentera_webrtc_ros_msgs::PeerStatusConstPtr &msg);
    void openteraEventCallback(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr &msg);

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
    ros::Subscriber m_openteraEventSubscriber;

};


#endif // MAINWINDOW_H
