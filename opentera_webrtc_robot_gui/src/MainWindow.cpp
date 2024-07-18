#include "MainWindow.h"
#include <QGraphicsScene>
#include <QThread>
#include <QDebug>
#include "utils.h"

constexpr bool NO_REPAINT = false;

MainWindow::MainWindow(QString devicePropertiesPath, rclcpp::Node& node, QWidget* parent)
    : QMainWindow{parent},
      m_deviceProperties{devicePropertiesPath, node},
      m_inSession{false},
      m_node{node}
{
    m_ui.setupUi(this);

    // Resize to specified default size
    resize(m_deviceProperties.screenWidth, m_deviceProperties.screenHeight);

    // Buttons
    setupButtons();

    // ConfigDialog
    m_configDialog = new ConfigDialog(this);

    // Statistics
    m_statistics = new Statistics(this);

    // Create camera view
    m_cameraView = new ROSCameraView("Local", m_ui.imageWidget);

    m_ui.imageWidgetLayout->addWidget(m_cameraView);
    m_localCameraWindow = new LocalCameraWindow(this);

    // Setup ROS
    setupROS();

    // Connect signals/slot
    connect(this, &MainWindow::newLocalImage, this, &MainWindow::_onLocalImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerImage, this, &MainWindow::_onPeerImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerStatus, this, &MainWindow::_onPeerStatus, Qt::QueuedConnection);
    connect(this, &MainWindow::newRobotStatus, this, &MainWindow::_onRobotStatus, Qt::QueuedConnection);

    // Buttons
    connect(m_ui.hangUpButton, &QPushButton::clicked, this, &MainWindow::_onHangUpButtonClicked);
    connect(m_ui.configButton, &QPushButton::clicked, this, &MainWindow::_onConfigButtonClicked);
    connect(m_ui.cameraVisibilityButton, &QPushButton::clicked, this, &MainWindow::_onCameraVisibilityButtonClicked);
    connect(m_ui.batteryButton, &QToolButton::clicked, this, &MainWindow::_onBatteryButtonClicked);
    connect(m_ui.networkButton, &QToolButton::clicked, this, &MainWindow::_onNetworkButtonClicked);

    connect(m_ui.cropFaceButton, &QPushButton::clicked, this, &MainWindow::_onCropFaceButtonClicked);
    connect(m_ui.microphoneButton, &QPushButton::clicked, this, &MainWindow::_onMicrophoneButtonClicked);
    connect(m_ui.cameraButton, &QPushButton::clicked, this, &MainWindow::_onCameraButtonClicked);
    connect(
        m_ui.cameraButton,
        &QPushButton::toggled,
        m_cameraView,
        [this] { m_cameraView->setVisible(!m_ui.cameraButton->isChecked()); });
    connect(m_ui.speakerButton, &QPushButton::clicked, this, &MainWindow::_onSpeakerButtonClicked);
}

void MainWindow::setupROS()
{
    // Setup subscribers
    m_localImageSubscriber = m_node.create_subscription<sensor_msgs::msg::Image>(
        "/front_camera/image_raw",
        10,
        bind_this<sensor_msgs::msg::Image>(this, &MainWindow::localImageCallback));

    m_peerImageSubscriber = m_node.create_subscription<opentera_webrtc_ros_msgs::msg::PeerImage>(
        "/webrtc_image",
        10,
        bind_this<opentera_webrtc_ros_msgs::msg::PeerImage>(this, &MainWindow::peerImageCallback));

    m_peerStatusSubscriber = m_node.create_subscription<opentera_webrtc_ros_msgs::msg::PeerStatus>(
        "/webrtc_peer_status",
        10,
        bind_this<opentera_webrtc_ros_msgs::msg::PeerStatus>(this, &MainWindow::peerStatusCallback));

    m_robotStatusSubscriber = m_node.create_subscription<opentera_webrtc_ros_msgs::msg::RobotStatus>(
        "/robot_status",
        10,
        bind_this<opentera_webrtc_ros_msgs::msg::RobotStatus>(this, &MainWindow::robotStatusCallback));

    // Setup publishers
    m_enableFaceCroppingPublisher = m_node.create_publisher<std_msgs::msg::Bool>("enable_face_cropping", 1);

    m_micVolumePublisher = m_node.create_publisher<std_msgs::msg::Float32>("mic_volume", 1);

    m_enableCameraPublisher = m_node.create_publisher<std_msgs::msg::Bool>("enable_camera", 1);

    m_volumePublisher = m_node.create_publisher<std_msgs::msg::Float32>("volume", 1);

    m_callAllPublisher = m_node.create_publisher<std_msgs::msg::Empty>("call_all", 1);

    m_manageSessionPublisher = m_node.create_publisher<std_msgs::msg::String>("manage_session", 1);
}

void MainWindow::localImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    // WARNING THIS IS CALLED FROM ANOTHER THREAD (ROS SPINNER)
    // qDebug() << "localImageCallback thread" << QThread::currentThread();

    if (msg->encoding == "rgb8")
    {
        // Step #1 Transform ROS Image to QtImage
        QImage image(&msg->data[0], msg->width, msg->height, QImage::Format_RGB888);

        // Step #2 emit new signal with image
        emit newLocalImage(image.copy());
    }
    else if (msg->encoding == "bgr8")
    {
        // Step #1 Transform ROS Image to QtImage
        QImage image(&msg->data[0], msg->width, msg->height, QImage::Format_RGB888);

        // Step #2 emit new signal with image
        // Invert R & B here
        emit newLocalImage(image.rgbSwapped());
    }
    else
    {
        qDebug() << "Unhandled image encoding: " << QString::fromStdString(msg->encoding);
        RCLCPP_ERROR(m_node.get_logger(), "Unhandled image encoding: %s", msg->encoding.c_str());
    }
}

void MainWindow::_onLocalImage(const QImage& image)
{
    m_cameraView->setImage(image, NO_REPAINT);
    repaint();
    if (m_localCameraWindow->isVisible())
    {
        m_localCameraWindow->repaint();
    }
}


void MainWindow::_onPeerImage(const QString& id, const QString& name, const QImage& image)
{
    if (m_inSession)
    {
        if (m_remoteViews.empty())
        {
            setLocalCameraStyle(CameraStyle::window);
        }

        if (!m_remoteViews.contains(id))
        {
            ROSCameraView* camera = new ROSCameraView(name, nullptr);
            camera->setImage(image, NO_REPAINT);
            m_ui.imageWidgetLayout->addWidget(camera);
            m_remoteViews[id] = camera;
        }
        else
        {
            m_remoteViews[id]->setImage(image, NO_REPAINT);
            m_remoteViews[id]->setText(name);
        }
    }
}

void MainWindow::onPeerStatusClientConnected()
{
    m_ui.hangUpButton->setEnabled(true);
    m_ui.hangUpButton->setChecked(true);
    m_inSession = true;
}

void MainWindow::onPeerStatusClientDisconnected(const QString& id)
{
    if (m_remoteViews.contains(id))
    {
        m_remoteViews[id]->deleteLater();
        m_remoteViews.remove(id);

        if (m_remoteViews.empty())
        {
            m_inSession = false;

            // Put back full size self camera
            m_cameraView->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);

            m_ui.hangUpButton->setEnabled(false);
            m_ui.hangUpButton->setChecked(false);

            setLocalCameraStyle(CameraStyle::widget);
        }
    }
}

void MainWindow::_onPeerStatus(const QString& id, const QString& name, int status)
{
    switch (status)
    {
        case opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_CLIENT_CONNECTED:
            onPeerStatusClientConnected();
            break;

        case opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_CLIENT_DISCONNECTED:
            onPeerStatusClientDisconnected(id);
            break;

        case opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_REMOTE_STREAM_ADDED:
            break;

        case opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_REMOTE_STREAM_REMOVED:
            break;

        case opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_CLIENT_CONNECTION_FAILED:
            break;

        default:
            qWarning() << "Status not handled " << status;
            RCLCPP_WARN(m_node.get_logger(), "Status not handled : %i", status);
            break;
    }
}

void MainWindow::setImage(const QImage& image)
{
    m_cameraView->setImage(image);
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    QMainWindow::closeEvent(event);
    QApplication::quit();
}

void MainWindow::closeCameraWindow()
{
    if (m_localCameraWindow)
    {
        m_localCameraWindow->close();
    }
}

void MainWindow::peerImageCallback(const opentera_webrtc_ros_msgs::msg::PeerImage::ConstSharedPtr& msg)
{
    // Step #1 Transform ROS Image to QtImage
    QImage image(&msg->frame.data[0], msg->frame.width, msg->frame.height, QImage::Format_RGB888);

    // Step #2 Emit signal (will be handled in Qt main thread)
    // Image will be automatically deleted when required
    // Invert R & B here
    emit newPeerImage(
        QString::fromStdString(msg->sender.id),
        QString::fromStdString(msg->sender.name),
        image.rgbSwapped());
}

void MainWindow::peerStatusCallback(const opentera_webrtc_ros_msgs::msg::PeerStatus::ConstSharedPtr& msg)
{
    emit newPeerStatus(QString::fromStdString(msg->sender.id), QString::fromStdString(msg->sender.name), msg->status);
}

void MainWindow::robotStatusCallback(const opentera_webrtc_ros_msgs::msg::RobotStatus::ConstSharedPtr& msg)
{
    /*
        void newRobotStatus(bool charging, float battery_voltage, float battery_current, float battery_level,
                        float cpu_usage, float mem_usage, float disk_usage, const QString &wifi_network,
                        float wifi_strength, const QString &local_ip);
    */
    emit newRobotStatus(
        msg->is_charging,
        msg->battery_voltage,
        msg->battery_current,
        msg->battery_level,
        msg->cpu_usage,
        msg->mem_usage,
        msg->disk_usage,
        QString::fromStdString(msg->wifi_network),
        msg->wifi_strength,
        msg->upload_speed,
        msg->download_speed,
        QString::fromStdString(msg->local_ip),
        msg->mic_volume,
        msg->is_camera_on,
        msg->volume);
}

void MainWindow::_onRobotStatus(
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
    float volume)
{
    m_statistics->updateCharts(
        battery_voltage,
        battery_current,
        battery_level,
        cpu_usage,
        mem_usage,
        disk_usage,
        wifi_network,
        wifi_strength,
        upload_speed,
        download_speed,
        local_ip);
    setBatteryLevel(is_charging, battery_level);
    setNetworkStrength(wifi_strength);
    m_ui.cameraButton->setChecked(!is_camera_on);
    m_configDialog->setMicVolumeSliderValue(mic_volume * 100);
    if (mic_volume == 0)
    {
        m_ui.microphoneButton->setChecked(true);
    }
    else
    {
        m_ui.microphoneButton->setChecked(false);
    }
    m_configDialog->setVolumeSliderValue(volume * 100);
    if (volume == 0)
    {
        m_ui.speakerButton->setChecked(true);
    }
    else
    {
        m_ui.speakerButton->setChecked(false);
    }
}

void MainWindow::setBatteryLevel(bool isCharging, float batteryLevel)
{
    QIcon newIcon;
    QString text;
    text.setNum(batteryLevel);
    if (isCharging)
    {
        newIcon.addFile(":/battery-charging.png");
    }
    else if (batteryLevel <= 5 && batteryLevel >= -0.1)
    {
        newIcon.addFile(":/battery-almost-empty.png");
    }
    else if (batteryLevel <= 33)
    {
        newIcon.addFile(":/battery-low.png");
    }
    else if (batteryLevel <= 66)
    {
        newIcon.addFile(":/battery-medium.png");
    }
    else if (batteryLevel <= 100)
    {
        newIcon.addFile(":/battery-full.png");
    }
    else
    {
        newIcon.addFile(":/battery-empty.png");
    }
    m_ui.batteryButton->setIcon(newIcon);
    text.append("%");
    m_ui.batteryButton->setText(text);
}

void MainWindow::setNetworkStrength(float wifiStrength)
{
    QIcon newIcon;
    if (wifiStrength <= 0.1)
    {
        newIcon.addFile(":/network-0-bars");
    }
    else if (wifiStrength <= 25)
    {
        newIcon.addFile(":/network-1-bar");
    }
    else if (wifiStrength <= 50)
    {
        newIcon.addFile(":/network-2-bars");
    }
    else if (wifiStrength <= 75)
    {
        newIcon.addFile(":/network-3-bars");
    }
    else
    {
        newIcon.addFile(":/network-4-bars");
    }
    m_ui.networkButton->setIcon(newIcon);
}

void MainWindow::setLocalCameraStyle(CameraStyle style)
{
    if (style != m_cameraView->getCurrentStyle())
    {
        if (style == CameraStyle::window)
        {
            m_ui.imageWidgetLayout->removeWidget(m_cameraView);
            m_localCameraWindow->addCamera(m_cameraView);
            m_cameraView->useWindowStyle();
            m_ui.cameraVisibilityButton->setVisible(true);
        }
        else
        {
            m_localCameraWindow->removeCamera(m_cameraView);
            m_ui.imageWidgetLayout->addWidget(m_cameraView);
            m_cameraView->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
            m_cameraView->useWidgetStyle();
            m_ui.cameraVisibilityButton->setVisible(false);
        }
    }
}

void MainWindow::setupButtons()
{
    QIcon phoneIcon;
    phoneIcon.addFile(QStringLiteral(":/phone-call-start.png"), QSize(), QIcon::Normal, QIcon::Off);
    phoneIcon.addFile(QStringLiteral(":/phone-call-end.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui.hangUpButton->setIcon(phoneIcon);
    m_ui.hangUpButton->setText("");

    m_ui.configButton->setIcon(QIcon(":/settings-gear.png"));
    m_ui.configButton->setText("");

    QIcon cropFaceIcon;
    cropFaceIcon.addFile(QStringLiteral(":/frame-person-disable.png"), QSize(), QIcon::Normal, QIcon::Off);
    cropFaceIcon.addFile(QStringLiteral(":/frame-person.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui.cropFaceButton->setIcon(cropFaceIcon);
    m_ui.cropFaceButton->setText("");

    QIcon cameraVisibilityIcon;
    cameraVisibilityIcon.addFile(QStringLiteral(":/hide-camera.png"), QSize(), QIcon::Normal, QIcon::Off);
    cameraVisibilityIcon.addFile(QStringLiteral(":/show-camera.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui.cameraVisibilityButton->setIcon(cameraVisibilityIcon);
    m_ui.cameraVisibilityButton->setText("");
    m_ui.cameraVisibilityButton->setVisible(false);

    QIcon cameraIcon;
    cameraIcon.addFile(QStringLiteral(":/video-camera-on.png"), QSize(), QIcon::Normal, QIcon::Off);
    cameraIcon.addFile(QStringLiteral(":/video-camera-off.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui.cameraButton->setIcon(cameraIcon);
    m_ui.cameraButton->setText("");


    QIcon micIcon;
    micIcon.addFile(QStringLiteral(":/mic-on.png"), QSize(), QIcon::Normal, QIcon::Off);
    micIcon.addFile(QStringLiteral(":/mic-off.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui.microphoneButton->setIcon(micIcon);
    m_ui.microphoneButton->setText("");

    QIcon speakerIcon;
    speakerIcon.addFile(QStringLiteral(":/volume.png"), QSize(), QIcon::Normal, QIcon::Off);
    speakerIcon.addFile(QStringLiteral(":/volume-mute.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui.speakerButton->setIcon(speakerIcon);
    m_ui.speakerButton->setText("");

    m_ui.batteryButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_ui.batteryButton->setIcon(QIcon(":/battery-empty.png"));
    m_ui.batteryButton->setText("0%");

    m_ui.networkButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_ui.networkButton->setIcon(QIcon(":/network-0-bars"));
    m_ui.networkButton->setText("");
}

QRect MainWindow::getCameraSpace()
{
    int taskbarHeight = this->frameGeometry().height() - this->geometry().height();
    QRect camRect = m_ui.imageWidget->geometry();
    camRect.moveTo(camRect.x() + this->pos().x(), camRect.y() + this->pos().y() + taskbarHeight);
    return camRect;
}

void MainWindow::_onHangUpButtonClicked()
{
    if (m_ui.hangUpButton->isChecked())
    {
        std_msgs::msg::Empty msg;
        m_callAllPublisher->publish(msg);
    }
    else
    {
        endCall();
    }
}

void MainWindow::endCall()
{
    std_msgs::msg::String msg;
    msg.data = "stop";
    m_manageSessionPublisher->publish(msg);
    m_ui.hangUpButton->setEnabled(false);
}

void MainWindow::_onConfigButtonClicked()
{
    m_configDialog->exec();
}

void MainWindow::_onCameraVisibilityButtonClicked()
{
    m_localCameraWindow->setVisible(!m_ui.cameraVisibilityButton->isChecked());
    if (m_ui.cameraVisibilityButton->isChecked())
    {
        m_configDialog->setOpacitySliderValue(0);
    }
    else
    {
        m_configDialog->setOpacitySliderValue(100);
    }
}

void MainWindow::_onBatteryButtonClicked()
{
    m_statistics->setCurrentPage("battery");
    m_statistics->exec();
}

void MainWindow::_onNetworkButtonClicked()
{
    m_statistics->setCurrentPage("network");
    m_statistics->exec();
}

void MainWindow::_onCropFaceButtonClicked()
{
    std_msgs::msg::Bool msg;
    if (m_ui.cropFaceButton->isChecked())
    {
        msg.data = false;
    }
    else
    {
        msg.data = true;
    }
    m_enableFaceCroppingPublisher->publish(msg);
}

void MainWindow::_onMicrophoneButtonClicked()
{
    std_msgs::msg::Float32 msg;
    if (m_ui.microphoneButton->isChecked())
    {
        msg.data = 0;
        m_configDialog->setMicVolumeSliderValue(0);
    }
    else
    {
        msg.data = 1;
        m_configDialog->setMicVolumeSliderValue(100);
    }
    m_micVolumePublisher->publish(msg);
}

void MainWindow::_onCameraButtonClicked()
{
    std_msgs::msg::Bool msg;
    msg.data = !m_ui.cameraButton->isChecked();
    m_localCameraWindow->setVisible(
        !m_ui.cameraButton->isChecked() && !m_ui.cameraVisibilityButton->isChecked() &&
        m_cameraView->getCurrentStyle() == CameraStyle::window);
    m_ui.cameraVisibilityButton->setVisible(
        !m_ui.cameraButton->isChecked() && m_cameraView->getCurrentStyle() == CameraStyle::window);
    m_enableCameraPublisher->publish(msg);
}

void MainWindow::_onSpeakerButtonClicked()
{
    std_msgs::msg::Float32 msg;
    if (m_ui.speakerButton->isChecked())
    {
        msg.data = 0;
        m_configDialog->setVolumeSliderValue(0);
    }
    else
    {
        msg.data = 1;
        m_configDialog->setVolumeSliderValue(100);
    }
    m_volumePublisher->publish(msg);
}

void MainWindow::onMicVolumeSliderValueChanged()
{
    float value = m_configDialog->getMicVolumeSliderValue();
    if (value == 0)
    {
        m_ui.microphoneButton->setChecked(true);
    }
    else
    {
        m_ui.microphoneButton->setChecked(false);
    }
    std_msgs::msg::Float32 msg;
    msg.data = value / 100;
    m_micVolumePublisher->publish(msg);
}

void MainWindow::onOpacitySliderValueChanged()
{
    int value = m_configDialog->getOpacitySliderValue();
    m_localCameraWindow->setWindowOpacity(value / 100.0);
    if (value == 0)
    {
        m_ui.cameraVisibilityButton->setChecked(true);
    }
    else
    {
        m_ui.cameraVisibilityButton->setChecked(false);

        if (!m_localCameraWindow->isVisible() && !m_ui.cameraButton->isChecked())
        {
            m_localCameraWindow->setVisible(true);
        }
    }
}

void MainWindow::onVolumeSliderValueChanged()
{
    float value = m_configDialog->getVolumeSliderValue();
    if (value == 0)
    {
        m_ui.speakerButton->setChecked(true);
    }
    else
    {
        m_ui.speakerButton->setChecked(false);
    }
    std_msgs::msg::Float32 msg;
    msg.data = value / 100;
    m_volumePublisher->publish(msg);
}

void MainWindow::moveEvent(QMoveEvent* event)
{
    m_localCameraWindow->followMainWindow(event->pos() - event->oldPos());
    QMainWindow::moveEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    m_localCameraWindow->adjustPositionFromBottomLeft(event->oldSize(), event->size());
    QMainWindow::resizeEvent(event);
}
