#include "ui_MainWindow.h"
#include "MainWindow.h"
#include <QGraphicsScene>
#include <QThread>
#include <QDebug>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), m_ui(new Ui::MainWindow)
{
    m_ui->setupUi(this);

    // Buttons
    setupButtons();

    //ConfigDialog
    m_configDialog = new ConfigDialog(this);

    //Statistics
    m_Statistics = new Statistics(this);

    // Toolbar
    m_toolbar = new GraphicsViewToolbar(m_ui->toolboxWidget);

    // Create camera view
    m_cameraView = new ROSCameraView("Local", m_ui->imageWidget);

    m_ui->imageWidgetLayout->addWidget(m_cameraView);

    // Setup ROS
    setupROS();

    // Connect signals/slot
    connect(this, &MainWindow::newLocalImage, this, &MainWindow::_onLocalImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerImage, this, &MainWindow::_onPeerImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerStatus, this, &MainWindow::_onPeerStatus, Qt::QueuedConnection);
    connect(this, &MainWindow::newRobotStatus, this, &MainWindow::_onRobotStatus, Qt::QueuedConnection);

    // Buttons
    connect(m_ui->configButton, &QPushButton::clicked, this, &MainWindow::_onConfigButtonClicked);
    connect(m_ui->batteryButton, &QToolButton::clicked, this, &MainWindow::_onBatteryButtonClicked); 

    connect(m_ui->microphoneButton, &QPushButton::clicked, this, &MainWindow::_onMicrophoneButtonClicked);
    connect(m_ui->cameraButton, &QPushButton::clicked, this, &MainWindow::_onCameraButtonClicked);
    connect(m_ui->cameraButton, &QPushButton::toggled, m_cameraView, [this]{ m_cameraView->setVisible(!m_ui->cameraButton->isChecked()); });
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

    m_volumePublisher= m_nodeHandle.advertise<std_msgs::Float32>("volume", 1);
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
    // qDebug() << "_onLocalImage Current Thread " << QThread::currentThread();
    m_cameraView->setImage(image);
}


void MainWindow::_onPeerImage(const QString& id, const QString& name, const QImage& image)
{
    if (!m_remoteViews.contains(id))
    {
        ROSCameraView* camera = new ROSCameraView(name, nullptr);
        camera->setImage(image);
        m_ui->imageWidgetLayout->addWidget(camera);
        m_remoteViews[id] = camera;
        m_cameraView->setMaximumSize(320, 240);
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
                    // Put back full size self camera
                    m_cameraView->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
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
    const QString& local_ip,
    float mic_volume,
    bool is_camera_on,
    float volume)
{
    m_toolbar->setBatteryStatus(is_charging, battery_voltage, battery_current, battery_level);
    m_statistics->updateCharts(battery_voltage, battery_current, battery_level);
    setBatteryLevel(is_charging, battery_level);
    setNetworkStrength(wifi_strength);
    m_ui->cameraButton->setChecked(!is_camera_on);
    m_configDialog->setMicVolumeSliderValue(mic_volume * 100);
    if(mic_volume == 0) {
        m_ui->microphoneButton->setChecked(true);
    } else {
        m_ui->microphoneButton->setChecked(false);
    }
    m_configDialog->setVolumeSliderValue(volume * 100);
    if(volume == 0){
        m_ui->speakerButton->setChecked(true);
    } else {
        m_ui->speakerButton->setChecked(false);
    }

}

void MainWindow::setBatteryLevel(bool is_charging, float battery_level){
    QIcon newIcon;
    QString text;
    text.setNum(battery_level);
    if(is_charging)
    {
        newIcon.addFile(":/battery-charging.png");
    }
    else if(battery_level <= 5)
    {
        newIcon.addFile(":/battery-almost-empty.png");
    }
    else if(battery_level <= 33)
    {
        newIcon.addFile(":/battery-low.png");
    }
    else if(battery_level <= 66)
    {
        newIcon.addFile(":/battery-medium.png");
    }
    else if(battery_level <= 100)
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

void MainWindow::setNetworkStrength(float wifi_strength){
    QIcon newIcon;
    if(wifi_strength == 0){
        newIcon.addFile(":/network-0-bars");
    }
    else if(wifi_strength <= 25)
    {
        newIcon.addFile(":/network-1-bar");
    }
    else if(wifi_strength <= 50)
    {
        newIcon.addFile(":/network-2-bars");
    }
    else if(wifi_strength <= 75)
    {
        newIcon.addFile(":/network-3-bars");
    }
    else{
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
    m_ui->microphoneButton->setCheckable(true);;
    
    QIcon speakerIcon;
    speakerIcon.addFile(QStringLiteral(":/volume.png"), QSize(), QIcon::Normal, QIcon::Off);
    speakerIcon.addFile(QStringLiteral(":/volume-mute.png"), QSize(), QIcon::Normal, QIcon::On);
    m_ui->speakerButton->setIcon(speakerIcon);
    m_ui->speakerButton->setText("");
    m_ui->speakerButton->setCheckable(true);

    /*QFont batteryFont = m_ui->batteryButton->font();
    batteryFont.setPointSize(8);
    m_ui->batteryButton->setFont(batteryFont);*/
    m_ui->batteryButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_ui->batteryButton->setIcon(QIcon(":/battery-empty.png"));
    m_ui->batteryButton->setText("0%");

    m_ui->networkButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_ui->networkButton->setIcon(QIcon(":/network-0-bars"));
    m_ui->networkButton->setText("");
}

void MainWindow::_onConfigButtonClicked()
{
    m_configDialog->exec();
}

void MainWindow::_onBatteryButtonClicked()
{
    m_statistics->exec();
}

void MainWindow::_onMicrophoneButtonClicked()
{
    std_msgs::Float32 msg;
    if(m_ui->microphoneButton->isChecked()) {
        msg.data = 0;
        m_configDialog->setMicVolumeSliderValue(0);
    } else {
        msg.data = 1;
        m_configDialog->setMicVolumeSliderValue(100);
    }
    m_micVolumePublisher.publish(msg);
}

void MainWindow::_onCameraButtonClicked()
{
    std_msgs::Bool msg;
    msg.data = !m_ui->cameraButton->isChecked();
    m_enableCameraPublisher.publish(msg);
}

void MainWindow::_onSpeakerButtonClicked()
{
    std_msgs::Float32 msg;
    if(m_ui->speakerButton->isChecked()) {
        msg.data = 0;
        m_configDialog->setVolumeSliderValue(0);
    } else {
        msg.data = 1;
        m_configDialog->setVolumeSliderValue(100);
    }
    m_volumePublisher.publish(msg);
}

void MainWindow::_onMicVolumeSliderValueChanged()
{
    float value = m_configDialog->getMicVolumeSliderValue();
    if(value == 0){
        m_ui->microphoneButton->setChecked(true);
    } else {
        m_ui->microphoneButton->setChecked(false);
    }
    std_msgs::Float32 msg;
    msg.data = value/100;
    m_micVolumePublisher.publish(msg);
}

void MainWindow::_onVolumeSliderValueChanged()
{
    float value = m_configDialog->getVolumeSliderValue();
    if(value == 0){
        m_ui->speakerButton->setChecked(true);
    } else {
        m_ui->speakerButton->setChecked(false);
    }
    std_msgs::Float32 msg;
    msg.data = value/100;
    m_volumePublisher.publish(msg);
}
