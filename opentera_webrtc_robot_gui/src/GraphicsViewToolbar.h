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

public slots:
    void setBatteryStatus(bool is_charging, float battery_voltage, float battery_current, float battery_level);

private:

    QGraphicsScene *m_scene;
    QGraphicsTextItem *m_batteryTextItem;

    virtual void resizeEvent(QResizeEvent *event) override;

};


#endif
