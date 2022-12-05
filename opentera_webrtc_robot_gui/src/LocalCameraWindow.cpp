#include "LocalCameraWindow.h"
#include "MainWindow.h"

LocalCameraWindow::LocalCameraWindow(MainWindow* parent) : QDialog(parent)
{
    m_parent = parent;

    setWindowFlags(Qt::FramelessWindowHint | Qt::WindowTitleHint | Qt::Tool | Qt::Dialog);
    setAttribute(Qt::WA_DeleteOnClose);
    setAttribute(Qt::WA_QuitOnClose);
    setAttribute(Qt::WA_TranslucentBackground);
    setWindowOpacity(m_parent->m_defaultLocalCameraOpacity / 100);
    QVBoxLayout* layout = new QVBoxLayout();
    layout->setMargin(0);
    setLayout(layout);
    setVisible(false);
    setSizeGripEnabled(true);
    resize(m_parent->m_defaultLocalCameraWidth, m_parent->m_defaultLocalCameraHeight);
}

void LocalCameraWindow::addCamera(QWidget* cameraView)
{
    layout()->addWidget(cameraView);
    moveToDefaultPosition();

    setVisible(true);
}

void LocalCameraWindow::removeCamera(QWidget* cameraView)
{
    layout()->removeWidget(cameraView);
    setVisible(false);
}

void LocalCameraWindow::moveToDefaultPosition()
{
    QRect mainWindowRect = m_parent->getCameraSpace();
    int x = m_parent->m_defaultLocalCameraX;
    int y = m_parent->m_defaultLocalCameraY;
    if (x > 0 && y > 0)
    {
        move(mainWindowRect.left() + x, mainWindowRect.top() + y);
    }
    else if (x < 0 && y > 0)
    {
        move(mainWindowRect.right() - width() + x, mainWindowRect.top() + y);
    }
    else if (x > 0 && y < 0)
    {
        move(mainWindowRect.left() + x, mainWindowRect.bottom() - height() + y);
    }
    else
    {
        move(mainWindowRect.right() - width() + x, mainWindowRect.bottom() - height() + y);
    }
}

void LocalCameraWindow::adjustPositionFromBottomLeft(QSize oldWindowSize, QSize newWindowSize)
{
    QPoint destination(pos().x(), pos().y() + (newWindowSize.height() - oldWindowSize.height()));
    destination = adjustPositionToBorders(destination);
    move(destination);
}

void LocalCameraWindow::followMainWindow(QPoint positionDiff)
{
    move(pos() + positionDiff);
}

void LocalCameraWindow::mousePressEvent(QMouseEvent* event)
{
    m_pos = event->pos();
}

void LocalCameraWindow::mouseMoveEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        QPoint diff = event->pos() - m_pos;
        move(adjustPositionToBorders(pos() + diff));
    }
}

QPoint LocalCameraWindow::adjustPositionToBorders(QPoint pos)
{
    QRect mainWindowRect = m_parent->getCameraSpace();
    if (pos.x() < mainWindowRect.left())
    {
        pos.setX(mainWindowRect.left());
    }
    else if (pos.x() > mainWindowRect.right() - width())
    {
        pos.setX(mainWindowRect.right() - width());
    }

    if (pos.y() < mainWindowRect.top())
    {
        pos.setY(mainWindowRect.top());
    }
    else if (pos.y() > mainWindowRect.bottom() - height())
    {
        pos.setY(mainWindowRect.bottom() - height());
    }

    setMaximumWidth(mainWindowRect.right() - pos.x());
    setMaximumHeight(mainWindowRect.bottom() - pos.y());

    return pos;
}
