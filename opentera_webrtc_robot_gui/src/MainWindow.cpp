#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QGraphicsScene>

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

}

MainWindow::~MainWindow()
{
    delete m_ui;
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

