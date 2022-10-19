#ifndef _BATTERY_STATISTICS_H_
#define _BATTERY_STATISTICS_H_

#include <QDialog>
#include <QtCharts>
#include <QDateTimeAxis>
#include <QLineSeries>
#include <chrono>

using namespace QtCharts;


QT_BEGIN_NAMESPACE
    namespace Ui
    {
        class Statistics;
    }
QT_END_NAMESPACE


class Statistics : public QDialog
{
    Q_OBJECT

public:
    Statistics(QWidget* parent = nullptr);
    ~Statistics();

    void updateCharts(float battery_voltage, float battery_current, float battery_level);

protected:
    Ui::Statistics* m_ui;
private:
    QLineSeries* m_batteryLevelLineSeries;
    QChart* m_batteryLevelChart;
    QChartView* m_batteryLevelChartView;

    QLineSeries* m_batteryVoltageLineSeries;
    QChart* m_batteryVoltageChart;
    QChartView* m_batteryVoltageChartView;

    QLineSeries* m_batteryCurrentLineSeries;
    QChart* m_batteryCurrentChart;
    QChartView* m_batteryCurrentChartView;

    void setupCharts();
    void setDefaultChart(QLineSeries* series, QChart* chart, int yAxisMin, int yAxisMax);
};

#endif  //#define _BATTERY_STATISTICS_H_
