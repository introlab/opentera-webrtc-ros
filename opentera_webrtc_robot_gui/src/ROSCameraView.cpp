#include "ROSCameraView.h"

#include <QPainter>

CameraWidget::CameraWidget(QWidget* parent) : QWidget{parent} {}

void CameraWidget::setImage(const QImage& image, bool repaintNow)
{
    // Make sure we copy the image (could be deleted somewhere else)
    m_image = image.copy();

    if (repaintNow)
    {
        repaint();
    }
}

void CameraWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    painter.fillRect(rect(), QBrush(Qt::black));

    if (m_image.width() <= 0 || m_image.height() <= 0)
    {
        return;
    }

    float scale =
        std::min(static_cast<float>(width()) / static_cast<float>(m_image.width()),
            static_cast<float>(height()) / static_cast<float>(m_image.height()));
    int scaledWidth = static_cast<int>(scale * m_image.width());
    int scaledHeight = static_cast<int>(scale * m_image.height());
    int offsetX = std::max(0, (width() - scaledWidth) / 2);
    int offsetY = std::max(0, (height() - scaledHeight) / 2);

    painter.drawImage(QRect(offsetX, offsetY, scaledWidth, scaledHeight), m_image);
}

ROSCameraView::ROSCameraView(QWidget* parent)
    : QWidget{parent},
      m_layout{nullptr},
      m_label{nullptr},
      m_cameraWidget{nullptr}
{
    m_layout = new QVBoxLayout(this);

    // Label
    m_label = new QLabel(this);
    m_label->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    m_label->setMaximumHeight(25);
    m_layout->addWidget(m_label);

    // CameraWidget
    m_cameraWidget = new CameraWidget(this);
    m_layout->addWidget(m_cameraWidget);

    m_currentStyle = CameraStyle::widget;
    m_widgetStyleLayout = m_layout;
}

ROSCameraView::ROSCameraView(const QString& label, QWidget* parent) : ROSCameraView{parent}
{
    m_label->setText(label);
}

void ROSCameraView::setText(const QString& text)
{
    if (m_label)
        m_label->setText(text);
}

void ROSCameraView::setImage(const QImage& image, bool repaintNow)
{
    if (m_cameraWidget)
        m_cameraWidget->setImage(image, repaintNow);
}

void ROSCameraView::useWindowStyle()
{
    if (m_currentStyle == CameraStyle::widget)
    {
        m_layout->setMargin(0);
        m_layout->setSpacing(0);
        m_layout->setContentsMargins(0, 0, 0, 0);
        m_layout->removeWidget(m_label);
        m_currentStyle = CameraStyle::window;
    }
}

void ROSCameraView::useWidgetStyle()
{
    if (m_currentStyle == CameraStyle::window)
    {
        m_layout = m_widgetStyleLayout;
        m_currentStyle = CameraStyle::widget;
        m_layout->insertWidget(0, m_label);
    }
}

CameraStyle ROSCameraView::getCurrentStyle()
{
    return m_currentStyle;
}
