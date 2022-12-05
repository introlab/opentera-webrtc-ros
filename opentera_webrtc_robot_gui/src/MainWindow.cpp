#include "ui_MainWindow.h"
#include "MainWindow.h"
#include <QGraphicsScene>
#include <QThread>
#include <QDebug>

MainWindow::MainWindow(QString devicePropertiesPath, QWidget* parent) : QMainWindow(parent), m_ui(new Ui::MainWindow)
{
    m_ui->setupUi(this);

    // Device properties
    setDeviceProperties(devicePropertiesPath);

    // Buttons
    setupButtons();

    // ConfigDialog
    m_configDialog = new ConfigDialog(this);

    // Statistics
    m_statistics = new Statistics(this);

    // Create camera view
    m_cameraView = new ROSCameraView("Local", m_ui->imageWidget);

    m_ui->imageWidgetLayout->addWidget(m_cameraView);
    m_localCameraWindow = new LocalCameraWindow(this);

    // Setup ROS
    setupROS();

    // Connect signals/slot
    connect(this, &MainWindow::newLocalImage, this, &MainWindow::_onLocalImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerImage, this, &MainWindow::_onPeerImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerStatus, this, &MainWindow::_onPeerStatus, Qt::QueuedConnection);
    connect(this, &MainWindow::newRobotStatus, this, &MainWindow::_onRobotStatus, Qt::QueuedConnection);

    // Buttons
    connect(m_ui->configButton, &QPushButton::clicked, this, &MainWindow::_onConfigButtonClicked);
    connect(m_ui->cameraVisibilityButton, &QPushButton::clicked, this, &MainWindow::_onCameraVisibilityButtonClicked);
    connect(m_ui->batteryButton, &QToolButton::clicked, this, &MainWindow::_onBatteryButtonClicked);
    connect(m_ui->networkButton, &QToolButton::clicked, this, &MainWindow::_onNetworkButtonClicked);

    connect(m_ui->microphoneButton, &QPushButton::clicked, this, &MainWindow::_onMicrophoneButtonClicked);
    connect(m_ui->cameraButton, &QPushButton::clicked, this, &MainWindow::_onCameraButtonClicked);
    connect(
        m_ui->cameraButton,
        &QPushButton::toggled,
        m_cameraView,
        [this] { m_cameraView->setVisible(!m_ui->cameraButton->isChecked()); });
    connect(m_ui->speakerButton, &QPushButton::clicked, this, &MainWindow::_onSpeakerButtonClicked);

    // Signaling events
    connect(this, &MainWindow::eventJoinSession, this, &MainWindow::_onJoinSessionEvent, Qt::QueuedConnection);
    connect(this, &MainWindow::eventLeaveSession, this, &MainWindow::_onLeaveSessionEvent, Qt::QueuedConnection);
    connect(this, &MainWindow::eventStopSession, this, &MainWindow::_onStopSessionEvent, Qt::QueuedConnection);
}

MainWindow::~MainWindow()
{
    delete m_ui;
}

void MainWindow::setupROS()
{
    // Setup subscribers
    m_localImageSubscriber =
        m_nodeHandle.subscribe("/front_camera/image_raw", 10, &MainWindow::localImageCallback, this);


    m_peerImageSubscriber = m_nodeHandle.subscribe("/webrtc_image", 10, &MainWindow::peerImageCallback, this);

    m_peerStatusSubscriber = m_nodeHandle.subscribe("/webrtc_peer_status", 10, &MainWindow::peerStatusCallback, this);


    m_openteraEventSubscriber = m_nodeHandle.subscribe("/events", 10, &MainWindow::openteraEventCallback, this);

    m_robotStatusSubscriber = m_nodeHandle.subscribe("/robot_status", 10, &MainWindow::robotStatusCallback, this);

    // Setup publishers
    m_micVolumePublisher = m_nodeHandle.advertise<std_msgs::Float32>("mic_volume", 1);

    m_enableCameraPublisher = m_nodeHandle.advertise<std_msgs::Bool>("enable_camera", 1);

    m_volumePublisher = m_nodeHandle.advertise<std_msgs::Float32>("volume", 1);
}

void MainWindow::setDeviceProperties(QString path)
{
    QFile file;
    file.setFileName(path);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QString properties = file.readAll();
        file.close();
        QJsonDocument doc = QJsonDocument::fromJson(properties.toUtf8());
        QJsonObject propertiesObject = doc.object();

        m_screenWidth = propertiesObject.value("width").toInt();
        m_screenHeight = propertiesObject.value("height").toInt();
        m_defaultLocalCameraWidth = propertiesObject.value("defaultLocalCameraWidth").toInt();
        m_defaultLocalCameraHeight = propertiesObject.value("defaultLocalCameraHeight").toInt();
        m_diagonalLength = propertiesObject.value("diagonalLength").toDouble();
        m_defaultLocalCameraOpacity = propertiesObject.value("defaultLocalCameraOpacity").toDouble();
        m_defaultLocalCameraX = propertiesObject.value("defaultLocalCameraX").toInt();
        m_defaultLocalCameraY = propertiesObject.value("defaultLocalCameraY").toInt();

        resize(m_screenWidth, m_screenHeight);
    }
    else
    {
        ROS_WARN("Device properties file not found, using default properties");

        m_screenWidth = 1080;
        m_screenHeight = 1920;
        m_defaultLocalCameraWidth = 300;
        m_defaultLocalCameraHeight = 200;
        m_diagonalLength = 15;
        m_defaultLocalCameraOpacity = 1;
        m_defaultLocalCameraX = 10;
        m_defaultLocalCameraY = -10;
    }
}

void MainWindow::localImageCallback(const sensor_msgs::ImageConstPtr& msg)
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
        ROS_ERROR("Unhandled image encoding: %s", msg->encoding.c_str());
    }
}

void MainWindow::openteraEventCallback(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr& msg)
{
    // WARNING THIS IS CALLED FROM ANOTHER THREAD (ROS SPINNER)

    // We are only interested in JoinSession, StopSession, LeaveSession events for now
    for (auto i = 0; i < msg->join_session_events.size(); i++)
    {
        QList<QString> session_participants;
        for (auto j = 0; j < msg->join_session_events[i].session_participants.size(); j++)
        {
            session_participants.append(QString::fromStdString(msg->join_session_events[i].session_participants[j]));
        }

        QList<QString> session_users;
        for (auto j = 0; j < msg->join_session_events[i].session_users.size(); j++)
        {
            session_users.append(QString::fromStdString(msg->join_session_events[i].session_users[j]));
        }

        QList<QString> session_devices;
        for (auto j = 0; j < msg->join_session_events[i].session_devices.size(); j++)
        {
            session_devices.append(QString::fromStdString(msg->join_session_events[i].session_devices[j]));
        }

        emit eventJoinSession(
            QString::fromStdString(msg->join_session_events[i].session_url),
            QString::fromStdString(msg->join_session_events[i].session_creator_name),
            QString::fromStdString(msg->join_session_events[i].session_uuid),
            session_participants,
            session_users,
            session_devices,
            QString::fromStdString(msg->join_session_events[i].join_msg),
            QString::fromStdString(msg->join_session_events[i].session_parameters),
            QString::fromStdString(msg->join_session_events[i].service_uuid));
    }

    for (auto i = 0; i < msg->leave_session_events.size(); i++)
    {
        QList<QString> leaving_participants;
        for (auto j = 0; j < msg->leave_session_events[i].leaving_participants.size(); j++)
        {
            leaving_participants.append(QString::fromStdString(msg->leave_session_events[i].leaving_participants[j]));
        }

        QList<QString> leaving_users;
        for (auto j = 0; j < msg->leave_session_events[i].leaving_users.size(); j++)
        {
            leaving_users.append(QString::fromStdString(msg->leave_session_events[i].leaving_users[j]));
        }

        QList<QString> leaving_devices;
        for (auto j = 0; j < msg->leave_session_events[i].leaving_devices.size(); j++)
        {
            leaving_devices.append(QString::fromStdString(msg->leave_session_events[i].leaving_devices[j]));
        }

        emit eventLeaveSession(
            QString::fromStdString(msg->leave_session_events[i].session_uuid),
            QString::fromStdString(msg->leave_session_events[i].service_uuid),
            leaving_participants,
            leaving_users,
            leaving_devices);
    }

    for (auto i = 0; i < msg->stop_session_events.size(); i++)
    {
        emit eventStopSession(
            QString::fromStdString(msg->stop_session_events[i].session_uuid),
            QString::fromStdString(msg->stop_session_events[i].service_uuid));
    }
}

void MainWindow::_onLocalImage(const QImage& image)
{
    m_cameraView->setImage(image);
}


void MainWindow::_onPeerImage(const QString& id, const QString& name, const QImage& image)
{
    if (m_remoteViews.empty())
    {
        m_ui->imageWidgetLayout->removeWidget(m_cameraView);
        m_localCameraWindow->addCamera(m_cameraView);
        m_cameraView->useWindowStyle();
        m_ui->cameraVisibilityButton->setVisible(true);
    }

    if (!m_remoteViews.contains(id))
    {
        ROSCameraView* camera = new ROSCameraView(name, nullptr);
        camera->setImage(image);
        m_ui->imageWidgetLayout->addWidget(camera);
        m_remoteViews[id] = camera;
    }
    else
    {
        m_remoteViews[id]->setImage(image);
        m_remoteViews[id]->setText(name);
    }
}

void MainWindow::_onPeerStatus(const QString& id, const QString& name, int status)
{
    switch (status)
    {
        case opentera_webrtc_ros_msgs::PeerStatus::STATUS_CLIENT_CONNECTED:
            break;

        case opentera_webrtc_ros_msgs::PeerStatus::STATUS_CLIENT_DISCONNECTED:
            if (m_remoteViews.contains(id))
            {
                m_remoteViews[id]->deleteLater();
                m_remoteViews.remove(id);

                if (m_remoteViews.empty())
                {
                    m_localCameraWindow->removeCamera(m_cameraView);
                    // Put back full size self camera
                    m_ui->imageWidgetLayout->addWidget(m_cameraView);
                    m_cameraView->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
                    m_cameraView->useWidgetStyle();
                    m_ui->cameraVisibilityButton->setVisible(false);
                }
            }
            break;

        case opentera_webrtc_ros_msgs::PeerStatus::STATUS_REMOTE_STREAM_ADDED:
            break;

        case opentera_webrtc_ros_msgs::PeerStatus::STATUS_REMOTE_STREAM_REMOVED:
            break;

        default:
            qWarning() << "Status not handled " << status;
            ROS_WARN("Status not handled : %i", status);
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
        delete m_localCameraWindow;
    }
}

void MainWindow::peerImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg)
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

void MainWindow::peerStatusCallback(const opentera_webrtc_ros_msgs::PeerStatusConstPtr& msg)
{
    emit newPeerStatus(QString::fromStdString(msg->sender.id), QString::fromStdString(msg->sender.name), msg->status);
}

void MainWindow::robotStatusCallback(const opentera_webrtc_ros_msgs::RobotStatusConstPtr& msg)
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


void MainWindow::_onJoinSessionEvent(
    const QString& session_url,
    const QString& session_creator_name,
    const QString& session_uuid,
    QList<QString> session_participants,
    QList<QString> session_users,
    QList<QString> session_devices,
    const QString& join_msg,
    const QString& session_parameters,
    const QString& service_uuid)
{
}

void MainWindow::_onStopSessionEvent(const QString& session_uuid, const QString& service_uuid)
{
    qDebug() << "_onStopSessionEvent(const QString &session_uuid, const QString &service_uuid)";
    ROS_DEBUG("_onStopSessionEvent(const QString &session_uuid, const QString &service_uuid)");

    // Remove all remote views
    foreach (QString key, m_remoteViews.keys())
    {
        m_remoteViews[key]->deleteLater();
    }

    m_remoteViews.clear();

    // Put back full size self camera
    m_cameraView->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
}

void MainWindow::_onLeaveSessionEvent(
    const QString& session_uuid,
    const QString& service_uuid,
    QList<QString> leaving_participants,
    QList<QString> leaving_users,
    QList<QString> leaving_devices)
{
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
    m_ui->cameraButton->setChecked(!is_camera_on);
    m_configDialog->setMicVolumeSliderValue(mic_volume * 100);
    if (mic_volume == 0)
    {
        m_ui->microphoneButton->setChecked(true);
    }
    else
    {
        m_ui->microphoneButton->setChecked(false);
    }
    m_configDialog->setVolumeSliderValue(volume * 100);
    if (volume == 0)
    {
        m_ui->speakerButton->setChecked(true);
    }
    else
    {
        m_ui->speakerButton->setChecked(false);
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
    m_ui->batteryButton->setIcon(newIcon);
    text.append("%");
    m_ui->batteryButton->setText(text);
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
    m_ui->networkButton->setIcon(newIcon);
}

void MainWindow::setupButtons()
{
    m_ui->hangUpButton->setIcon(QIcon(":/phone-call-end.png"));
    m_ui->hangUpButton->setText("");

    m_ui->configButton->setIcon(QIcon(":/settings-gear.png"));
    m_ui->configButton->setText("");

    QIcon cameraVisibilityIcon;
    cameraVisibilityIcon.addFile(QStringLiteral(":/hide-camera.png"), QSize(), QIcon::Normal, QIcon::Off);
    cameraVisibilityIcon.addFile(QStringLiteral(":/show-camera.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui->cameraVisibilityButton->setIcon(cameraVisibilityIcon);
    m_ui->cameraVisibilityButton->setText("");
    m_ui->cameraVisibilityButton->setVisible(false);

    QIcon cameraIcon;
    cameraIcon.addFile(QStringLiteral(":/video-camera-on.png"), QSize(), QIcon::Normal, QIcon::Off);
    cameraIcon.addFile(QStringLiteral(":/video-camera-off.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui->cameraButton->setIcon(cameraIcon);
    m_ui->cameraButton->setText("");
    m_ui->cameraButton->setCheckable(true);


    QIcon micIcon;
    micIcon.addFile(QStringLiteral(":/mic-on.png"), QSize(), QIcon::Normal, QIcon::Off);
    micIcon.addFile(QStringLiteral(":/mic-off.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui->microphoneButton->setIcon(micIcon);
    m_ui->microphoneButton->setText("");
    m_ui->microphoneButton->setCheckable(true);

    QIcon speakerIcon;
    speakerIcon.addFile(QStringLiteral(":/volume.png"), QSize(), QIcon::Normal, QIcon::Off);
    speakerIcon.addFile(QStringLiteral(":/volume-mute.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui->speakerButton->setIcon(speakerIcon);
    m_ui->speakerButton->setText("");
    m_ui->speakerButton->setCheckable(true);

    m_ui->batteryButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_ui->batteryButton->setIcon(QIcon(":/battery-empty.png"));
    m_ui->batteryButton->setText("0%");

    m_ui->networkButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_ui->networkButton->setIcon(QIcon(":/network-0-bars"));
    m_ui->networkButton->setText("");
}

QRect MainWindow::getCameraSpace()
{
    int taskbarHeight = this->frameGeometry().height() - this->geometry().height();
    QRect camRect = m_ui->imageWidget->geometry();
    camRect.moveTo(camRect.x() + this->pos().x(), camRect.y() + this->pos().y() + taskbarHeight);
    return camRect;
}

void MainWindow::_onConfigButtonClicked()
{
    m_configDialog->exec();
}

void MainWindow::_onCameraVisibilityButtonClicked()
{
    m_localCameraWindow->setVisible(!m_ui->cameraVisibilityButton->isChecked());
    if (m_ui->cameraVisibilityButton->isChecked())
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

void MainWindow::_onMicrophoneButtonClicked()
{
    std_msgs::Float32 msg;
    if (m_ui->microphoneButton->isChecked())
    {
        msg.data = 0;
        m_configDialog->setMicVolumeSliderValue(0);
    }
    else
    {
        msg.data = 1;
        m_configDialog->setMicVolumeSliderValue(100);
    }
    m_micVolumePublisher.publish(msg);
}

void MainWindow::_onCameraButtonClicked()
{
    std_msgs::Bool msg;
    msg.data = !m_ui->cameraButton->isChecked();
    m_localCameraWindow->setVisible(!m_ui->cameraButton->isChecked() && !m_ui->cameraVisibilityButton->isChecked());
    m_ui->cameraVisibilityButton->setVisible(!m_ui->cameraButton->isChecked());
    m_enableCameraPublisher.publish(msg);
}

void MainWindow::_onSpeakerButtonClicked()
{
    std_msgs::Float32 msg;
    if (m_ui->speakerButton->isChecked())
    {
        msg.data = 0;
        m_configDialog->setVolumeSliderValue(0);
    }
    else
    {
        msg.data = 1;
        m_configDialog->setVolumeSliderValue(100);
    }
    m_volumePublisher.publish(msg);
}

void MainWindow::onMicVolumeSliderValueChanged()
{
    float value = m_configDialog->getMicVolumeSliderValue();
    if (value == 0)
    {
        m_ui->microphoneButton->setChecked(true);
    }
    else
    {
        m_ui->microphoneButton->setChecked(false);
    }
    std_msgs::Float32 msg;
    msg.data = value / 100;
    m_micVolumePublisher.publish(msg);
}

void MainWindow::onOpacitySliderValueChanged()
{
    float value = m_configDialog->getOpacitySliderValue();
    m_localCameraWindow->setWindowOpacity(value / 100);
    if (value == 0)
    {
        m_ui->cameraVisibilityButton->setChecked(true);
    }
    else
    {
        m_ui->cameraVisibilityButton->setChecked(false);

        if (!m_localCameraWindow->isVisible() && !m_ui->cameraButton->isChecked())
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
        m_ui->speakerButton->setChecked(true);
    }
    else
    {
        m_ui->speakerButton->setChecked(false);
    }
    std_msgs::Float32 msg;
    msg.data = value / 100;
    m_volumePublisher.publish(msg);
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
