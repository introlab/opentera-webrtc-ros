#include "Statistics.h"
#include "MainWindow.h"

#include "ui_Statistics.h"

Statistics::Statistics(QWidget* parent) : m_ui(new Ui::Statistics())
{
    m_ui->setupUi(this);
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint | Qt::WindowTitleHint);
    setupCharts();
}

Statistics::~Statistics() {}

void Statistics::setupCharts() 
{
    m_batteryLevelLineSeries = new QLineSeries();
    m_batteryLevelChart = new QChart();
    setDefaultChart(m_batteryLevelLineSeries, m_batteryLevelChart, 0, 100);
    m_batteryLevelChartView = new QChartView(m_batteryLevelChart);

    m_batteryVoltageLineSeries = new QLineSeries();
    m_batteryVoltageChart = new QChart();
    setDefaultChart(m_batteryVoltageLineSeries, m_batteryVoltageChart, 0, 25);
    m_batteryVoltageChartView = new QChartView(m_batteryVoltageChart);


    m_batteryCurrentLineSeries = new QLineSeries();
    m_batteryCurrentChart = new QChart();
    setDefaultChart(m_batteryCurrentLineSeries, m_batteryCurrentChart, 0, 5);
    m_batteryCurrentChartView = new QChartView(m_batteryCurrentChart);

    m_ui->firstGroupBox->layout()->addWidget(m_batteryLevelChartView);
    m_ui->secondGroupBox->layout()->addWidget(m_batteryVoltageChartView);
    m_ui->thirdGroupBox->layout()->addWidget(m_batteryCurrentChartView);
}

void Statistics::setDefaultChart(QLineSeries* series, QChart* chart, int yAxisMin, int yAxisMax){
    chart->legend()->hide();
    chart->addSeries(series);

    chart->setMargins(QMargins(2, 2, 2, 2));
    chart->setTheme(QChart::ChartThemeDark);
    chart->setBackgroundBrush(QBrush(QColor(0, 0, 0, 127)));

    QDateTimeAxis* xAxis = new QDateTimeAxis();
    QValueAxis* yAxis = new QValueAxis();

    yAxis->setRange(yAxisMin,yAxisMax);
    yAxis->setTitleText("");
    xAxis->setTitleText("");

    chart->addAxis(xAxis, Qt::AlignBottom);
    chart->addAxis(yAxis, Qt::AlignLeft);

    series->attachAxis(xAxis);
    series->attachAxis(yAxis);

    xAxis->setMin(QDateTime::currentDateTime());
}

void Statistics::updateCharts(float battery_voltage, float battery_current, float battery_level)
{
    QDateTime now = QDateTime::currentDateTime(); 

    QDateTimeAxis* levelXAxis = (QDateTimeAxis*)m_batteryLevelChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* voltageXAxis = (QDateTimeAxis*)m_batteryVoltageChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* currentXAxis = (QDateTimeAxis*)m_batteryCurrentChart->axes(Qt::Horizontal)[0]; 

    m_batteryLevelLineSeries->append(now.toMSecsSinceEpoch(), battery_level);
    levelXAxis->setMax(now);

    m_batteryVoltageLineSeries->append(now.toMSecsSinceEpoch(), battery_voltage);
    voltageXAxis->setMax(now);

    m_batteryCurrentLineSeries->append(now.toMSecsSinceEpoch(), battery_current);
    currentXAxis->setMax(now);

    //Order is important, setting the format after setting the max and the min prevents a visual bug where the axis disappears 
    if(levelXAxis->format() != "h:mm")
    {
        /*levelXAxis->setMin(now);
        voltageXAxis->setMin(now);
        currentXAxis->setMin(now);*/

        levelXAxis->setFormat("h:mm");
        voltageXAxis->setFormat("h:mm");
        currentXAxis->setFormat("h:mm");
    } 
}
