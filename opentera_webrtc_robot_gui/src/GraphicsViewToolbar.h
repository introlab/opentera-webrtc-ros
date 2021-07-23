#ifndef _GRAPHICS_VIEW_TOOLBAR_H_
#define _GRAPHICS_VIEW_TOOLBAR_H_

#include <QGraphicsView>
#include <QGraphicsTextItem>
#include <QGraphicsEllipseItem>
#include <QGraphicsRectItem>
#include <QGraphicsPixmapItem>

class GraphicsViewToolbar : public QGraphicsView
{
    Q_OBJECT
public:
    GraphicsViewToolbar(QWidget *parent=nullptr);
private:

    QGraphicsScene *m_scene;
    QGraphicsTextItem *m_textItem;

    virtual void resizeEvent(QResizeEvent *event) override;

};


#endif
