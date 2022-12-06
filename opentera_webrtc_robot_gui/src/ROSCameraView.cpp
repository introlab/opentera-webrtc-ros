#include "ROSCameraView.h"
#include <QSize>
#include <QBrush>
#include <QColor>
#include <QDebug>

GLCameraWidget::GLCameraWidget(QWidget* parent) : QGLWidget(parent)
{
    m_layout = new QVBoxLayout(this);
    setLayout(m_layout);
}

void GLCameraWidget::setImage(const QImage& image)
{
    // Make sure we copy the image (could be deleted somewhere else)
    m_image = image.copy();
    update();
}


void GLCameraWidget::paintEvent(QPaintEvent* event)
{
    QPainter p(this);

    // Set the painter to use a smooth scaling algorithm.
    p.setRenderHint(QPainter::SmoothPixmapTransform, 1);

    // Draw Black Background
    p.fillRect(this->rect(), QBrush(Qt::black));


    if (m_image.width() > 0 && m_image.height() > 0)
    {
        // Find minimal scale
        float scale = std::min((float)this->width() / m_image.width(), (float)this->height() / m_image.height());
        float new_width = scale * m_image.width();
        float new_height = scale * m_image.height();
        int offset_x = (rect().width() - new_width) / 2;
        int offset_y = (rect().height() - new_height) / 2;

        // Draw image
        QRect drawingRect(std::max(0, offset_x), std::max(0, offset_y), new_width, new_height);

        // Paint in current rect
        p.drawImage(drawingRect, m_image);

        // This will strech the image...
        // p.drawImage(this->rect(), m_image);
    }
}

ROSCameraView::ROSCameraView(QWidget* parent)
    : QWidget(parent),
      m_layout(nullptr),
      m_label(nullptr),
      m_cameraWidget(nullptr)
{
    m_layout = new QVBoxLayout(this);

    // Label
    m_label = new QLabel(this);
    m_label->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    m_label->setMaximumHeight(25);
    m_layout->addWidget(m_label);

    // CameraWidget
    m_cameraWidget = new GLCameraWidget(this);
    m_layout->addWidget(m_cameraWidget);

    m_widgetStyleLayout = m_layout;
}

ROSCameraView::ROSCameraView(const QString& label, QWidget* parent) : ROSCameraView(parent)
{
    m_label->setText(label);
}

void ROSCameraView::setText(const QString& text)
{
    if (m_label)
        m_label->setText(text);
}

void ROSCameraView::setImage(const QImage& image)
{
    if (m_cameraWidget)
        m_cameraWidget->setImage(image);
}

void ROSCameraView::useWindowStyle()
{
    if (m_layout == m_widgetStyleLayout)
    {
        m_layout->setMargin(0);
        m_layout->setSpacing(0);
        m_layout->setContentsMargins(0, 0, 0, 0);
        m_layout->removeWidget(m_label);
    }
}

void ROSCameraView::useWidgetStyle()
{
    m_layout = m_widgetStyleLayout;
    m_layout->insertWidget(0, m_label);
}
