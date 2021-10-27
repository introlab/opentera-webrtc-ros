#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QGraphicsScene>
#include <QThread>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
{
    m_ui->setupUi(this);

    //Toolbar
    m_toolbar = new GraphicsViewToolbar(m_ui->toolboxWidget);


    //Create camera view
    m_cameraView = new ROSCameraView("Local", m_ui->imageWidget);


    m_ui->imageWidgetLayout->addWidget(m_cameraView);

    //Setup ROS
    setupROS();

    //Connect signals/slot
    connect(this, &MainWindow::newLocalImage, this, &MainWindow::_onLocalImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerImage, this, &MainWindow::_onPeerImage, Qt::QueuedConnection);
    connect(this, &MainWindow::newPeerStatus, this, &MainWindow::_onPeerStatus, Qt::QueuedConnection);
}

MainWindow::~MainWindow()
{
    delete m_ui;
}

void MainWindow::setupROS()
{
    //Setup subscribers
    m_localImageSubscriber = m_nodeHandle.subscribe("/front_camera/image_raw",
            10,
            &MainWindow::localImageCallback,
            this);

    m_peerImageSubscriber = m_nodeHandle.subscribe("/webrtc_image",
            10,
            &MainWindow::peerImageCallback,
            this);

    m_peerStatusSubscriber = m_nodeHandle.subscribe("/webrtc_peer_status",
            10, 
            &MainWindow::peerStatusCallback,
            this);

    //Setup publishers
}

void MainWindow::localImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    //WARNING THIS IS CALLED FROM ANOTHER THREAD (ROS SPINNER)
    //qDebug() << "localImageCallback thread" << QThread::currentThread();
    
    //Step #1 Transform ROS Image to QtImage
    QImage image(&msg->data[0], msg->width, msg->height, QImage::Format_RGB888);
    
    //Step #2 Emit signal (will be handled in Qt main thread)
    //Image will be automatically deleted when required
    
    //emit newLocalImage(image);
    //Invert R & B here
    emit newLocalImage(std::move(image).rgbSwapped());
}



void MainWindow::_onLocalImage(const QImage& image)
{
    //qDebug() << "_onLocalImage Current Thread " << QThread::currentThread();
    m_cameraView->setImage(image);
}


void MainWindow::_onPeerImage(const QString &id, const QString &name, const QImage& image)
{
    if (!m_remoteViews.contains(id))
    {
        ROSCameraView *camera = new ROSCameraView(name, m_ui->imageWidget);
        camera->setImage(image);
        m_ui->imageWidgetLayout->addWidget(camera);
        m_remoteViews[id] = camera;

        m_cameraView->setMaximumSize(320,240);
    }
    else
    {
        m_remoteViews[id]->setImage(image);
        m_remoteViews[id]->setText(name);
    }
}

void MainWindow::_onPeerStatus(const QString &id, const QString& name, int status)
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
                //Put back full size self camera
                m_cameraView->setMaximumSize(QWIDGETSIZE_MAX,QWIDGETSIZE_MAX);
            }
        }
        break;

        case opentera_webrtc_ros_msgs::PeerStatus::STATUS_REMOTE_STREAM_ADDED:
        break;

        case opentera_webrtc_ros_msgs::PeerStatus::STATUS_REMOTE_STREAM_REMOVED:
        break;

        default:
            qWarning() << "status not hanelded " << status;
        break;
    }
}

void MainWindow::setImage(const QImage &image)
{
    m_cameraView->setImage(image);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);
    QApplication::quit();
}

void MainWindow::peerImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr &msg)
{
    //Step #1 Transform ROS Image to QtImage
    QImage image(&msg->frame.data[0], msg->frame.width, msg->frame.height, QImage::Format_RGB888);
    
    //Step #2 Emit signal (will be handled in Qt main thread)
    //Image will be automatically deleted when required
    //Invert R & B here
    emit newPeerImage(QString::fromStdString(msg->sender.id), QString::fromStdString(msg->sender.name), image.rgbSwapped());
}

void MainWindow::peerStatusCallback(const opentera_webrtc_ros_msgs::PeerStatusConstPtr &msg)
{
    emit newPeerStatus(QString::fromStdString(msg->sender.id), QString::fromStdString(msg->sender.name), msg->status);
}
