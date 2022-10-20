#include "Statistics.h"
#include "MainWindow.h"

#include "ui_Statistics.h"

Statistics::Statistics(QWidget* parent) : m_ui(new Ui::Statistics())
{
    m_ui->setupUi(this);
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint | Qt::WindowTitleHint);
    setupMenu();
    setupCharts();
}

Statistics::~Statistics() {}

void Statistics::setupMenu()
{   
    m_ui->menuButton->setIcon(QIcon(":/three-horizontal-lines.png"));

    m_ui->menuBarFrame->setVisible(false);
    connect(m_ui->menuButton, &QPushButton::clicked, this, &Statistics::_onMenuButtonClicked);

    m_ui->batteryButton->setIcon(QIcon(":/battery-full.png"));
    m_ui->batteryButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    connect(m_ui->batteryButton, &QPushButton::clicked, this, [this]{setCurrentPage("battery");});
    
    m_ui->networkButton->setIcon(QIcon(":/network-4-bars.png"));
    m_ui->batteryButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    connect(m_ui->networkButton, &QPushButton::clicked, this, [this]{setCurrentPage("network");});
    
    m_ui->systemButton->setIcon(QIcon(":/computer.png"));
    m_ui->batteryButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    connect(m_ui->systemButton, &QPushButton::clicked, this, [this]{setCurrentPage("system");});
}

void Statistics::setupCharts() 
{
    m_batteryLevelLineSeries = new QLineSeries();
    m_batteryLevelChart = new QChart();
    setDefaultChart(m_batteryLevelLineSeries, m_batteryLevelChart, 0, 100);

    m_batteryVoltageLineSeries = new QLineSeries();
    m_batteryVoltageChart = new QChart();
    setDefaultChart(m_batteryVoltageLineSeries, m_batteryVoltageChart, 0, 25);


    m_batteryCurrentLineSeries = new QLineSeries();
    m_batteryCurrentChart = new QChart();
    setDefaultChart(m_batteryCurrentLineSeries, m_batteryCurrentChart, 0, 5);

    m_networkStrengthLineSeries = new QLineSeries();
    m_networkStrengthChart = new QChart();
    setDefaultChart(m_networkStrengthLineSeries, m_networkStrengthChart, 0, 100);

    m_cpuUsageLineSeries = new QLineSeries();
    m_cpuUsageChart = new QChart();
    setDefaultChart(m_cpuUsageLineSeries, m_cpuUsageChart, 0, 100);

    m_memUsageLineSeries = new QLineSeries();
    m_memUsageChart = new QChart();
    setDefaultChart(m_memUsageLineSeries, m_memUsageChart, 0, 100);

    m_diskUsageLineSeries = new QLineSeries();
    m_diskUsageChart = new QChart();
    setDefaultChart(m_diskUsageLineSeries, m_diskUsageChart, 0, 100);

    
    m_firstChartView = new QChartView(m_batteryLevelChart);
    m_secondChartView = new QChartView(m_batteryVoltageChart);
    m_thirdChartView = new QChartView(m_batteryCurrentChart);
    m_ui->firstGroupBox->layout()->addWidget(m_firstChartView);
    m_ui->secondGroupBox->layout()->addWidget(m_secondChartView);
    m_ui->thirdGroupBox->layout()->addWidget(m_thirdChartView);
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

void Statistics::updateCharts(
    float battery_voltage, 
    float battery_current, 
    float battery_level,
    float cpu_usage,
    float mem_usage,
    float disk_usage,
    //const QString& wifi_network, TODO i
    float wifi_strength
    //const QString& local_ip, TODO i
    )
{ //TODO i https://stackoverflow.com/questions/57748558/how-to-scroll-view-when-plot-reaches-a-qchartview-border
    QDateTime now = QDateTime::currentDateTime(); 

    QDateTimeAxis* batteryLevelXAxis = (QDateTimeAxis*)m_batteryLevelChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* batteryVoltageXAxis = (QDateTimeAxis*)m_batteryVoltageChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* batteryCurrentXAxis = (QDateTimeAxis*)m_batteryCurrentChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* networkStrengthXAxis = (QDateTimeAxis*)m_networkStrengthChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* cpuUsageXAxis = (QDateTimeAxis*)m_cpuUsageChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* memUsageXAxis = (QDateTimeAxis*)m_memUsageChart->axes(Qt::Horizontal)[0]; 
    QDateTimeAxis* diskUsageXAxis = (QDateTimeAxis*)m_diskUsageChart->axes(Qt::Horizontal)[0]; 

    m_batteryLevelLineSeries->append(now.toMSecsSinceEpoch(), battery_level);
    batteryLevelXAxis->setMax(now);

    m_batteryVoltageLineSeries->append(now.toMSecsSinceEpoch(), battery_voltage);
    batteryVoltageXAxis->setMax(now);

    m_batteryCurrentLineSeries->append(now.toMSecsSinceEpoch(), battery_current);
    batteryCurrentXAxis->setMax(now);

    m_networkStrengthLineSeries->append(now.toMSecsSinceEpoch(), wifi_strength);
    networkStrengthXAxis->setMax(now);

    m_cpuUsageLineSeries->append(now.toMSecsSinceEpoch(), cpu_usage);
    cpuUsageXAxis->setMax(now);

    m_memUsageLineSeries->append(now.toMSecsSinceEpoch(), mem_usage);
    memUsageXAxis->setMax(now);

    m_diskUsageLineSeries->append(now.toMSecsSinceEpoch(), disk_usage);
    diskUsageXAxis->setMax(now);

    //Order is important, setting the format once after setting the max and the min prevents a visual bug where the axis disappears 
    if(batteryLevelXAxis->format() != "h:mm")
    {
        /*levelXAxis->setMin(now); TODO i
        voltageXAxis->setMin(now);
        currentXAxis->setMin(now);*/

        batteryLevelXAxis->setFormat("h:mm");
        batteryVoltageXAxis->setFormat("h:mm");
        batteryCurrentXAxis->setFormat("h:mm");
        networkStrengthXAxis->setFormat("h:mm");
        cpuUsageXAxis->setFormat("h:mm");
        memUsageXAxis->setFormat("h:mm");
        diskUsageXAxis->setFormat("h:mm");
    } 
}

void Statistics::setCurrentPage(QString page)
{
    if(page ==  "battery")
    {
        m_ui->menuLabel->setText("Battery");
        m_ui->firstGroupBox->setTitle("Charge (%)");
        m_firstChartView->setChart(m_batteryLevelChart);
        m_ui->secondGroupBox->setTitle("Voltage (V)");
        m_secondChartView->setChart(m_batteryVoltageChart);
        m_ui->thirdGroupBox->setTitle("Current (A)");
        m_thirdChartView->setChart(m_batteryCurrentChart);

        m_ui->infoFrame->setVisible(false);
        m_ui->secondGroupBox->setVisible(true);
        m_ui->thirdGroupBox->setVisible(true);
    }
    else if(page == "network")
    {
        m_ui->menuLabel->setText("Network");
        m_ui->firstInfoLabel->setText("todo wifi name");
        m_ui->secondInfoLabel->setText("todo ip");
        m_ui->firstGroupBox->setTitle("Wifi strength (?)");
        m_firstChartView->setChart(m_networkStrengthChart);

        m_ui->infoFrame->setVisible(true);
        m_ui->secondGroupBox->setVisible(false);
        m_ui->thirdGroupBox->setVisible(false);
    }
    else if(page == "system")
    {
        m_ui->menuLabel->setText("System");
        m_ui->firstGroupBox->setTitle("CPU (%)");
        m_firstChartView->setChart(m_cpuUsageChart);
        m_ui->secondGroupBox->setTitle("Memory (%)");
        m_secondChartView->setChart(m_memUsageChart);
        m_ui->thirdGroupBox->setTitle("Storage (%)");
        m_thirdChartView->setChart(m_diskUsageChart);

        m_ui->infoFrame->setVisible(false);
        m_ui->secondGroupBox->setVisible(true);
        m_ui->thirdGroupBox->setVisible(true);
    }
}

void Statistics::_onMenuButtonClicked() //TODO i oneline?
{
    m_ui->menuBarFrame->setVisible(!m_ui->menuBarFrame->isVisible());
}