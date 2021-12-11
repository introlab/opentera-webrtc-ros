#include "GraphicsViewToolbar.h"
#include <QResizeEvent>
#include <QDebug>
#include <QLayout>

GraphicsViewToolbar::GraphicsViewToolbar(QWidget *parent)
  :  QGraphicsView(parent)
{
    m_scene = new QGraphicsScene(this);

    QFont f;
    f.setPixelSize(12);
    f.setBold(true);

    m_batteryTextItem = new QGraphicsTextItem(nullptr);
    m_batteryTextItem->setPlainText("Battery");
    m_batteryTextItem->setFont(f);
    m_batteryTextItem->setPos(0,0);
    m_batteryTextItem->adjustSize();
    m_batteryTextItem->setDefaultTextColor(Qt::red);
    m_batteryTextItem->show();
    m_batteryTextItem->setZValue(1);

    m_scene->addItem(m_batteryTextItem);

    setScene(m_scene);

    show();

    if (parent && parent->layout())
        parent->layout()->addWidget(this);

    this->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void GraphicsViewToolbar::resizeEvent(QResizeEvent *event)
{
    //qDebug() << "resizeEvent" << event;

    m_scene->setSceneRect(0,0,event->size().width(), this->height());
    //resize(event->size());

    QGraphicsView::resizeEvent(event);
}

void GraphicsViewToolbar::setBatteryStatus(bool is_charging, float battery_voltage,
    float battery_current, float battery_level)
{
    m_batteryTextItem->setPlainText(QString("C: %1, %2 V, %3 A L:%4 %")
        .arg(is_charging)
        .arg(battery_voltage)
        .arg(battery_current)
        .arg(battery_level));

    m_batteryTextItem->adjustSize();

}