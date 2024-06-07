#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ui_MainWindow.h"
#include "Statistics.h"
#include "ConfigDialog.h"
#include "ROSCameraView.h"
#include "LocalCameraWindow.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_image.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_status.hpp>
#include <opentera_webrtc_ros_msgs/msg/open_tera_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/robot_status.hpp>
#include <QImage>
#include <QSharedPointer>
#include <QMap>
#include <QToolButton>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>


struct DeviceProperties
{
    int screenWidth = 600;
    int screenHeight = 1024;
    int defaultLocalCameraWidth = 320;
    int defaultLocalCameraHeight = 240;
    double diagonalLength = 7;
    double defaultLocalCameraOpacity = 90;
    int defaultLocalCameraX = 10;
    int defaultLocalCameraY = -10;

    explicit DeviceProperties(QString jsonFilePath, rclcpp::Node& node)
    {
        QFile file;
        file.setFileName(jsonFilePath);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QString properties = file.readAll();
            file.close();
            QJsonDocument doc = QJsonDocument::fromJson(properties.toUtf8());
            QJsonObject propertiesObject = doc.object();

            screenWidth = propertiesObject.value("width").toInt(screenWidth);
            screenHeight = propertiesObject.value("height").toInt(screenHeight);
            defaultLocalCameraWidth = propertiesObject.value("defaultLocalCameraWidth").toInt(defaultLocalCameraWidth);
            defaultLocalCameraHeight =
                propertiesObject.value("defaultLocalCameraHeight").toInt(defaultLocalCameraHeight);
            diagonalLength = propertiesObject.value("diagonalLength").toDouble(diagonalLength);
            defaultLocalCameraOpacity =
                propertiesObject.value("defaultLocalCameraOpacity").toDouble(defaultLocalCameraOpacity);
            defaultLocalCameraX = propertiesObject.value("defaultLocalCameraX").toInt(defaultLocalCameraX);
            defaultLocalCameraY = propertiesObject.value("defaultLocalCameraY").toInt(defaultLocalCameraY);
        }
        else
        {
            RCLCPP_WARN(node.get_logger(), "Device properties file not found, using default properties");
        }
    }
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QString devicePropertiesPath, rclcpp::Node& node, QWidget* parent = nullptr);
    ~MainWindow() = default;
    void setImage(const QImage& image);
    QRect getCameraSpace();
    void onMicVolumeSliderValueChanged();
    void onVolumeSliderValueChanged();
    void endCall();
    void onOpacitySliderValueChanged();
    void closeCameraWindow();
    DeviceProperties m_deviceProperties;

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

    void _onHangUpButtonClicked();
    void _onConfigButtonClicked();
    void _onCameraVisibilityButtonClicked();
    void _onBatteryButtonClicked();
    void _onNetworkButtonClicked();
    void _onCropFaceButtonClicked();
    void _onMicrophoneButtonClicked();
    void _onCameraButtonClicked();
    void _onSpeakerButtonClicked();

private:
    void setupROS();
    void setupButtons();
    void setBatteryLevel(bool isCharging, float batteryLevel);
    void setNetworkStrength(float wifiStrength);
    void setLocalCameraStyle(CameraStyle style);
    void closeEvent(QCloseEvent* event) override;
    void moveEvent(QMoveEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

    bool m_inSession;

    // Main View
    Ui::MainWindow m_ui;

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
    void localImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void peerImageCallback(const opentera_webrtc_ros_msgs::msg::PeerImage::ConstSharedPtr& msg);
    void peerStatusCallback(const opentera_webrtc_ros_msgs::msg::PeerStatus::ConstSharedPtr& msg);
    void openteraEventCallback(const opentera_webrtc_ros_msgs::msg::OpenTeraEvent::ConstSharedPtr& msg);
    void robotStatusCallback(const opentera_webrtc_ros_msgs::msg::RobotStatus::ConstSharedPtr& msg);

    rclcpp::Node& m_node;
    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::PeerImage>::SharedPtr m_peerImageSubscriber;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_localImageSubscriber;
    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::PeerStatus>::SharedPtr m_peerStatusSubscriber;
    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::OpenTeraEvent>::SharedPtr m_openteraEventSubscriber;
    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::RobotStatus>::SharedPtr m_robotStatusSubscriber;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr m_enableFaceCroppingPublisher;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr m_micVolumePublisher;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr m_enableCameraPublisher;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr m_volumePublisher;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr m_callAllPublisher;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_manageSessionPublisher;
};


#endif  // MAINWINDOW_H
