#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "Statistics.h"
#include "ConfigDialog.h"
#include "ROSCameraView.h"
#include "LocalCameraWindow.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>
#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>
#include <opentera_webrtc_ros_msgs/RobotStatus.h>
#include <QImage>
#include <QSharedPointer>
#include <QMap>
#include <QToolButton>
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
    MainWindow(QString devicePropertiesPath, QWidget* parent = nullptr);
    ~MainWindow();
    void setImage(const QImage& image);
    QRect getCameraSpace();
    void onMicVolumeSliderValueChanged();
    void onVolumeSliderValueChanged();
    void onOpacitySliderValueChanged();
    void closeCameraWindow();

    // Device properties
    int m_screenWidth;
    int m_screenHeight;
    int m_defaultLocalCameraWidth;
    int m_defaultLocalCameraHeight;
    double m_diagonalLength;
    double m_defaultLocalCameraOpacity;
    int m_defaultLocalCameraX;
    int m_defaultLocalCameraY;

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
        float upload_speed,
        float download_speed,
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
        float upload_speed,
        float download_speed,
        const QString& local_ip,
        float mic_volume,
        bool is_camera_on,
        float volume);

    void _onConfigButtonClicked();
    void _onCameraVisibilityButtonClicked();
    void _onBatteryButtonClicked();
    void _onNetworkButtonClicked();
    void _onMicrophoneButtonClicked();
    void _onCameraButtonClicked();
    void _onSpeakerButtonClicked();

private:
    void setupROS();
    void setupButtons();
    void setDeviceProperties(QString path);
    void setBatteryLevel(bool isCharging, float batteryLevel);
    void setNetworkStrength(float wifiStrength);
    void closeEvent(QCloseEvent* event) override;
    void moveEvent(QMoveEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;


    // Main View
    Ui::MainWindow* m_ui;

    // Local camera
    ROSCameraView* m_cameraView;
    LocalCameraWindow* m_localCameraWindow;

    // ConfigDialog
    ConfigDialog* m_configDialog;

    // Statistics
    Statistics* m_statistics;

    // Remote views
    QMap<QString, ROSCameraView*> m_remoteViews;

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
