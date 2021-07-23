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

    m_textItem = new QGraphicsTextItem(nullptr);
    m_textItem->setPlainText("Hello World!");
    m_textItem->setFont(f);
    m_textItem->setPos(0,0);
    m_textItem->adjustSize();
    m_textItem->setDefaultTextColor(Qt::red);
    m_textItem->show();
    m_textItem->setZValue(1);

    m_scene->addItem(m_textItem);


    QGraphicsRectItem *rect = new QGraphicsRectItem(10,10,50,20);
    rect->show();
    m_scene->addItem(rect);
    setScene(m_scene);

    m_scene->addRect(10,10,50,50,QPen(Qt::black));
    show();

    if (parent && parent->layout())
        parent->layout()->addWidget(this);

}

void GraphicsViewToolbar::resizeEvent(QResizeEvent *event)
{
    qDebug() << "resizeEvent" << event;

    m_scene->setSceneRect(0,0,event->size().width(), this->height());
    //resize(event->size());

    QGraphicsView::resizeEvent(event);
}
