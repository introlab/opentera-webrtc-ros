#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ConfigDialog.h"
#include "GraphicsViewToolbar.h"
#include "ROSCameraView.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>
#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>
#include <opentera_webrtc_ros_msgs/RobotStatus.h>
#include <QImage>
#include <QSharedPointer>
#include <QMap>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

QT_BEGIN_NAMESPACE
    namespace Ui
    {
        class MainWindow;
    }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();
    void setImage(const QImage& image);
    void onMicVolumeSliderValueChanged();
    void onVolumeSliderValueChanged();


signals:
    void newLocalImage(const QImage& image);
    void newPeerImage(const QString& id, const QString& name, const QImage& image);
    void newPeerStatus(const QString& id, const QString& name, int status);
    void newRobotStatus(
        bool is_charging,
        float battery_voltage,
        float battery_current,
        float battery_level,
        float cpu_usage,
        float mem_usage,
        float disk_usage,
        const QString& wifi_network,
        float wifi_strength,
        const QString& local_ip,
        float mic_volume,
        bool is_camera_on,
        float volume);
    void eventJoinSession(
        const QString& session_url,
        const QString& session_creator_name,
        const QString& session_uuid,
        QList<QString> session_participants,
        QList<QString> session_users,
        QList<QString> session_devices,
        const QString& join_msg,
        const QString& session_parameters,
        const QString& service_uuid);

    void eventStopSession(const QString& session_uuid, const QString& service_uuid);

    void eventLeaveSession(
        const QString& session_uuid,
        const QString& service_uuid,
        QList<QString> leaving_participants,
        QList<QString> leaving_users,
        QList<QString> leaving_devices);

private slots:
    void _onLocalImage(const QImage& image);
    void _onPeerImage(const QString& id, const QString& name, const QImage& image);
    void _onPeerStatus(const QString& id, const QString& name, int status);

    void _onJoinSessionEvent(
        const QString& session_url,
        const QString& session_creator_name,
        const QString& session_uuid,
        QList<QString> session_participants,
        QList<QString> session_users,
        QList<QString> session_devices,
        const QString& join_msg,
        const QString& session_parameters,
        const QString& service_uuid);

    void _onStopSessionEvent(const QString& session_uuid, const QString& service_uuid);

    void _onLeaveSessionEvent(
        const QString& session_uuid,
        const QString& service_uuid,
        QList<QString> leaving_participants,
        QList<QString> leaving_users,
        QList<QString> leaving_devices);

    void _onRobotStatus(
        bool is_charging,
        float battery_voltage,
        float battery_current,
        float battery_level,
        float cpu_usage,
        float mem_usage,
        float disk_usage,
        const QString& wif_network,
        float wifi_strength,
        const QString& local_ip,
        float mic_volume,
        bool is_camera_on,
        float volume);

    void _onConfigButtonClicked();
    void _onMicrophoneButtonClicked();
    void _onCameraButtonClicked();
    void _onSpeakerButtonClicked();

private:
    void setupROS();
    void setupButtons();
    void closeEvent(QCloseEvent* event) override;

    Ui::MainWindow* m_ui;

    // ConfigDialog
    ConfigDialog* m_configDialog;

    // Toolbar
    GraphicsViewToolbar* m_toolbar;

    // Remote views
    QMap<QString, ROSCameraView*> m_remoteViews;

    // Main View
    ROSCameraView* m_cameraView;

    // ROS

    // ROS Callbacks
    void localImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void peerImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg);
    void peerStatusCallback(const opentera_webrtc_ros_msgs::PeerStatusConstPtr& msg);
    void openteraEventCallback(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr& msg);
    void robotStatusCallback(const opentera_webrtc_ros_msgs::RobotStatusConstPtr& msg);

    ros::NodeHandle m_nodeHandle;
    ros::Subscriber m_peerImageSubscriber;
    ros::Subscriber m_localImageSubscriber;
    ros::Subscriber m_peerStatusSubscriber;
    ros::Subscriber m_openteraEventSubscriber;
    ros::Subscriber m_robotStatusSubscriber;
    ros::Publisher m_micVolumePublisher;
    ros::Publisher m_enableCameraPublisher;
    ros::Publisher m_volumePublisher;
};


#endif  // MAINWINDOW_H
