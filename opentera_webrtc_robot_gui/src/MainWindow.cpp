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
    m_cameraView = new ROSCameraView("Local", m_ui->centralwidget);
    m_ui->verticalLayout->addWidget(m_cameraView);

    //Setup ROS
    setupROS();

    //Connect signals/slot
    connect(this, &MainWindow::newLocalImage, this, &MainWindow::_onLocalImage, Qt::QueuedConnection);
}

MainWindow::~MainWindow()
{
    delete m_ui;
}

void MainWindow::setupROS()
{
    //Setup subscribers
    m_localImageSubscriber = m_nodeHandle.subscribe("/camera1/image_raw",
            10,
            &MainWindow::localImageCallback,
            this);

    //m_peerImageSubscriber = 

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
    //Invert R & B here
    emit newLocalImage(image.rgbSwapped());
}

void MainWindow::_onLocalImage(const QImage& image)
{
    //qDebug() << "_onLocalImage Current Thread " << QThread::currentThread();
    m_cameraView->setImage(image);
}


void MainWindow::setImage(const QImage &image)
{
    m_cameraView->setImage(image);
}

ROSCameraView* MainWindow::addThumbnailView(QImage &image, const QString &label)
{
    ROSCameraView *camera = new ROSCameraView(label, m_ui->thumbnailWidget);
    camera->setImage(image);
    m_ui->thubnailHorizontalLayout->addWidget(camera);
    return camera;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);
    QApplication::quit();
}