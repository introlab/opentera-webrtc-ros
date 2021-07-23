#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "GraphicsViewToolbar.h"
#include "ROSCameraView.h"

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
    ROSCameraView* addThumbnailView(QImage &image, const QString &label="");

private:
    Ui::MainWindow *m_ui;
    //Toolbar
    GraphicsViewToolbar *m_toolbar;
    //Local view
    ROSCameraView *m_cameraView;
};
#endif // MAINWINDOW_H
