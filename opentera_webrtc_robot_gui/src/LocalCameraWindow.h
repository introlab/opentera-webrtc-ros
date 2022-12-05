#ifndef _LOCAL_CAMERA_WINDOW_H_
#define _LOCAL_CAMERA_WINDOW_H_

#include <QDialog>

class MainWindow;

class LocalCameraWindow : public QDialog
{
    Q_OBJECT

public:
    LocalCameraWindow(MainWindow* parent = nullptr);

    void addCamera(QWidget* cameraView);
    void removeCamera(QWidget* cameraView);
    void moveToDefaultPosition();
    void adjustPositionFromBottomLeft(QSize oldWindowSize, QSize newWindowSize);
    void followMainWindow(QPoint positionDiff);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

private:
    QPoint m_pos;
    MainWindow* m_parent;

    QPoint adjustPositionToBorders(QPoint pos);
};

#endif
