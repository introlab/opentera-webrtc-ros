#ifndef _ROS_CAMERA_VIEW_H_
#define _ROS_CAMERA_VIEW_H_

#include <QWidget>
#include <QImage>
#include <QPaintEvent>
#include <QLabel>
#include <QVBoxLayout>


class CameraWidget : public QWidget
{
    Q_OBJECT

public:
    CameraWidget(QWidget* parent = nullptr);

    void setImage(const QImage& image, bool repaintNow = true);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage m_image;
    QVBoxLayout* m_layout;
    QWidget* m_parent;
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
    void setImage(const QImage& image, bool repaintNow = true);
    void useWindowStyle();
    void useWidgetStyle();

private:
    CameraStyle m_currentStyle;

    QVBoxLayout* m_layout;
    QVBoxLayout* m_widgetStyleLayout;
    CameraWidget* m_cameraWidget;
    QLabel* m_label;
};


#endif
