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

    void updateCharts(
        float battery_voltage, 
        float battery_current, 
        float battery_level,
        float cpu_usage,
        float mem_usage,
        float disk_usage,
        const QString& wifi_network,
        float wifi_strength,
        float upload_speed,
        float download_speed,
        const QString& local_ip
    );

public slots:
    void setCurrentPage(QString page);
protected:
    Ui::Statistics* m_ui;
private slots:
    void _onMenuButtonClicked();
private:
    QDateTime startTime;
    int maxRange;

    int maxUploadSpeed = 0;
    int minUploadSpeed = 1000000000;

    int maxDownloadSpeed = 0;
    int minDownloadSpeed = 1000000000;

    QLineSeries* m_batteryLevelLineSeries;
    QChart* m_batteryLevelChart;

    QLineSeries* m_batteryVoltageLineSeries;
    QChart* m_batteryVoltageChart;

    QLineSeries* m_batteryCurrentLineSeries;
    QChart* m_batteryCurrentChart;

    QLineSeries* m_networkStrengthLineSeries;
    QChart* m_networkStrengthChart;

    QLineSeries* m_uploadSpeedLineSeries;
    QChart* m_uploadSpeedChart;

    QLineSeries* m_downloadSpeedLineSeries;
    QChart* m_downloadSpeedChart;

    QLineSeries* m_cpuUsageLineSeries;
    QChart* m_cpuUsageChart;

    QLineSeries* m_memUsageLineSeries;
    QChart* m_memUsageChart;

    QLineSeries* m_diskUsageLineSeries;
    QChart* m_diskUsageChart;

    QChartView* m_firstChartView;
    QChartView* m_secondChartView;
    QChartView* m_thirdChartView;

    void setupMenu();

    void setupCharts();
    void setDefaultChart(QLineSeries* series, QChart* chart, int yAxisMin, int yAxisMax);

    void adjustNetworkSpeedCharts(QDateTime now, float uploadSpeed, float downloadSpeed);
};

#endif  //#define _BATTERY_STATISTICS_H_
