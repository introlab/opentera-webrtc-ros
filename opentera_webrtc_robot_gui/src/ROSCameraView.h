#ifndef _ROS_CAMERA_VIEW_H_
#define _ROS_CAMERA_VIEW_H_

#include <QGLWidget>
#include <QImage>
#include <QPaintEvent>
#include <QLabel>
#include <QVBoxLayout>


class GLCameraWidget : public QGLWidget
{
    Q_OBJECT

public:
    GLCameraWidget(QWidget* parent = nullptr);

    void setImage(const QImage& image);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage m_image;
    QVBoxLayout* m_layout;
};


enum class CameraStyle
{
    window,
    widget
};


class ROSCameraView : public QWidget
{
    Q_OBJECT

public:
    ROSCameraView(QWidget* parent = nullptr);
    ROSCameraView(const QString& label, QWidget* parent = nullptr);

    CameraStyle getCurrentStyle();

public slots:
    void setText(const QString& text);
    void setImage(const QImage& image);
    void useWindowStyle();
    void useWidgetStyle();

private:
    CameraStyle m_currentStyle;

    QVBoxLayout* m_layout;
    QVBoxLayout* m_widgetStyleLayout;
    GLCameraWidget* m_cameraWidget;
    QLabel* m_label;
};


#endif
