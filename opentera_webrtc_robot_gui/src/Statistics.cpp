#include "Statistics.h"
#include "MainWindow.h"

Statistics::Statistics(MainWindow* parent)
{
    m_ui.setupUi(this);
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint | Qt::WindowTitleHint);
    setupMenu();
    setupCharts();
    startTime = QDateTime::currentDateTime();

    m_graphTab = new QTabWidget(this);
    if (parent->m_deviceProperties.diagonalLength <= 10)
    {
        resize(parent->size().width() * 0.9, parent->size().height() * 0.9);
        QGridLayout* layout = dynamic_cast<QGridLayout*>(m_ui.graphFrame->layout());
        layout->removeWidget(m_ui.firstGroupBox);
        layout->removeWidget(m_ui.secondGroupBox);
        layout->removeWidget(m_ui.thirdGroupBox);
        m_ui.graphHorizontalLayout->removeWidget(m_ui.graphFrame);

        m_graphTab->tabBar()->setDocumentMode(true);
        m_graphTab->tabBar()->setExpanding(true);
        m_graphTab->addTab(m_ui.firstGroupBox, m_ui.firstGroupBox->title());
        m_graphTab->addTab(m_ui.secondGroupBox, m_ui.secondGroupBox->title());
        m_graphTab->addTab(m_ui.thirdGroupBox, m_ui.thirdGroupBox->title());
        m_graphTab->setStyleSheet("QTabBar::tab { height: 50px; width: 160px; }");
        m_ui.graphHorizontalLayout->addWidget(m_graphTab);
    }
}

void Statistics::setupMenu()
{
    m_ui.menuBarFrame->setVisible(false);

    m_ui.menuButton->setIcon(QIcon(":/three-horizontal-lines.png"));
    connect(
        m_ui.menuButton,
        &QPushButton::clicked,
        this,
        [this] { m_ui.menuBarFrame->setVisible(!m_ui.menuBarFrame->isVisible()); });

    QFont titleFont;
    titleFont.setPointSize(15);
    titleFont.setBold(true);
    m_ui.menuLabel->setFont(titleFont);

    maxRange = 1800;
    m_ui.timeFrameComboBox->addItem("Last minute", 60);
    m_ui.timeFrameComboBox->addItem("Last 5 minutes", 300);
    m_ui.timeFrameComboBox->addItem("Last 15 minutes", 900);
    m_ui.timeFrameComboBox->addItem("Last 30 minutes", maxRange);

    m_ui.batteryMenuButton->setIcon(QIcon(":/battery-full.png"));
    m_ui.batteryMenuButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_ui.batteryMenuButton->setText("Battery");
    m_ui.batteryMenuButton->setCheckable(true);
    connect(m_ui.batteryMenuButton, &QPushButton::clicked, this, [this] { setCurrentPage("battery"); });

    m_ui.networkMenuButton->setIcon(QIcon(":/network-4-bars.png"));
    m_ui.networkMenuButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_ui.networkMenuButton->setText("Network");
    m_ui.networkMenuButton->setCheckable(true);
    connect(m_ui.networkMenuButton, &QPushButton::clicked, this, [this] { setCurrentPage("network"); });

    m_ui.systemMenuButton->setIcon(QIcon(":/computer.png"));
    m_ui.systemMenuButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_ui.systemMenuButton->setText("System");
    m_ui.systemMenuButton->setCheckable(true);
    connect(m_ui.systemMenuButton, &QPushButton::clicked, this, [this] { setCurrentPage("system"); });
}

void Statistics::setupCharts()
{
    QFont infoFont;
    infoFont.setPointSize(15);
    infoFont.setBold(true);
    m_ui.firstInfoLabel->setFont(infoFont);
    m_ui.secondInfoLabel->setFont(infoFont);

    QFont groupBoxFont;
    groupBoxFont.setPointSize(12);
    groupBoxFont.setBold(true);
    m_ui.firstGroupBox->setFont(groupBoxFont);
    m_ui.secondGroupBox->setFont(groupBoxFont);
    m_ui.thirdGroupBox->setFont(groupBoxFont);

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

    m_uploadSpeedLineSeries = new QLineSeries();
    m_uploadSpeedChart = new QChart();
    setDefaultChart(m_uploadSpeedLineSeries, m_uploadSpeedChart, 0, 300);

    m_downloadSpeedLineSeries = new QLineSeries();
    m_downloadSpeedChart = new QChart();
    setDefaultChart(m_downloadSpeedLineSeries, m_downloadSpeedChart, 0, 300);

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
    m_ui.firstGroupBox->layout()->addWidget(m_firstChartView);
    m_ui.secondGroupBox->layout()->addWidget(m_secondChartView);
    m_ui.thirdGroupBox->layout()->addWidget(m_thirdChartView);
}

void Statistics::setDefaultChart(QLineSeries* series, QChart* chart, int yAxisMin, int yAxisMax)
{
    chart->legend()->hide();
    chart->addSeries(series);

    chart->setMargins(QMargins(2, 2, 2, 2));
    chart->setTheme(QChart::ChartThemeDark);
    chart->setBackgroundBrush(QBrush(QColor(0, 0, 0, 127)));

    QDateTimeAxis* xAxis = new QDateTimeAxis();
    QValueAxis* yAxis = new QValueAxis();

    yAxis->setRange(yAxisMin, yAxisMax);
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
    const QString& wifi_network,
    float wifi_strength,
    float upload_speed,
    float download_speed,
    const QString& local_ip)
{
    QDateTime now = QDateTime::currentDateTime();

    QDateTimeAxis* batteryLevelXAxis = static_cast<QDateTimeAxis*>(m_batteryLevelChart->axes(Qt::Horizontal)[0]);
    QDateTimeAxis* batteryVoltageXAxis = static_cast<QDateTimeAxis*>(m_batteryVoltageChart->axes(Qt::Horizontal)[0]);
    QDateTimeAxis* batteryCurrentXAxis = static_cast<QDateTimeAxis*>(m_batteryCurrentChart->axes(Qt::Horizontal)[0]);
    QDateTimeAxis* networkStrengthXAxis = static_cast<QDateTimeAxis*>(m_networkStrengthChart->axes(Qt::Horizontal)[0]);
    QDateTimeAxis* uploadSpeedXAxis = static_cast<QDateTimeAxis*>(m_uploadSpeedChart->axes(Qt::Horizontal)[0]);
    QDateTimeAxis* downloadSpeedXAxis = static_cast<QDateTimeAxis*>(m_downloadSpeedChart->axes(Qt::Horizontal)[0]);
    QValueAxis* uploadSpeedYAxis = static_cast<QValueAxis*>(m_uploadSpeedChart->axes(Qt::Vertical)[0]);
    QValueAxis* downloadSpeedYAxis = static_cast<QValueAxis*>(m_downloadSpeedChart->axes(Qt::Vertical)[0]);
    QDateTimeAxis* cpuUsageXAxis = static_cast<QDateTimeAxis*>(m_cpuUsageChart->axes(Qt::Horizontal)[0]);
    QDateTimeAxis* memUsageXAxis = static_cast<QDateTimeAxis*>(m_memUsageChart->axes(Qt::Horizontal)[0]);
    QDateTimeAxis* diskUsageXAxis = static_cast<QDateTimeAxis*>(m_diskUsageChart->axes(Qt::Horizontal)[0]);

    m_batteryLevelLineSeries->append(now.toMSecsSinceEpoch(), battery_level);
    m_batteryVoltageLineSeries->append(now.toMSecsSinceEpoch(), battery_voltage);
    m_batteryCurrentLineSeries->append(now.toMSecsSinceEpoch(), battery_current);

    QString networkName = wifi_network;
    networkName.replace(QString("\n"), QString(""));
    m_ui.firstInfoLabel->setText(networkName);
    m_ui.secondInfoLabel->setText(local_ip);

    m_networkStrengthLineSeries->append(now.toMSecsSinceEpoch(), wifi_strength);

    float uploadSpeedMb = upload_speed / 1048576;
    float downloadSpeedMb = download_speed / 1048576;

    m_uploadSpeedLineSeries->append(now.toMSecsSinceEpoch(), uploadSpeedMb);
    m_downloadSpeedLineSeries->append(now.toMSecsSinceEpoch(), downloadSpeedMb);

    // TODO change units and max according to the visible max value
    uploadSpeedYAxis->setMax(getMaxYAxisData(m_uploadSpeedLineSeries, now));
    downloadSpeedYAxis->setMax(getMaxYAxisData(m_downloadSpeedLineSeries, now));

    m_cpuUsageLineSeries->append(now.toMSecsSinceEpoch(), cpu_usage);
    m_memUsageLineSeries->append(now.toMSecsSinceEpoch(), mem_usage);
    m_diskUsageLineSeries->append(now.toMSecsSinceEpoch(), disk_usage);

    int range = m_ui.timeFrameComboBox->currentData().toInt();
    int elapsedTime = now.toSecsSinceEpoch() - startTime.toSecsSinceEpoch();
    if (elapsedTime < range)
    {
        batteryLevelXAxis->setMax(now);
        batteryVoltageXAxis->setMax(now);
        batteryCurrentXAxis->setMax(now);
        networkStrengthXAxis->setMax(now);
        uploadSpeedXAxis->setMax(now);
        downloadSpeedXAxis->setMax(now);
        cpuUsageXAxis->setMax(now);
        memUsageXAxis->setMax(now);
        diskUsageXAxis->setMax(now);
    }
    else
    {
        QDateTime minTime = now.addSecs(-1 * range);
        batteryLevelXAxis->setRange(minTime, now);
        batteryVoltageXAxis->setRange(minTime, now);
        batteryCurrentXAxis->setRange(minTime, now);
        networkStrengthXAxis->setRange(minTime, now);
        uploadSpeedXAxis->setRange(minTime, now);
        downloadSpeedXAxis->setRange(minTime, now);
        cpuUsageXAxis->setRange(minTime, now);
        memUsageXAxis->setRange(minTime, now);
        diskUsageXAxis->setRange(minTime, now);
    }

    if (elapsedTime >= maxRange)
    {
        m_batteryLevelLineSeries->remove(0);
        m_batteryVoltageLineSeries->remove(0);
        m_batteryCurrentLineSeries->remove(0);
        m_networkStrengthLineSeries->remove(0);
        m_uploadSpeedLineSeries->remove(0);
        m_downloadSpeedLineSeries->remove(0);
        m_cpuUsageLineSeries->remove(0);
        m_memUsageLineSeries->remove(0);
        m_diskUsageLineSeries->remove(0);
    }

    // Order is important
    // Setting the format once after setting the max and the min prevents a bug where the axis disappears
    if (m_ui.timeFrameComboBox->currentData().toInt() == 60 && batteryLevelXAxis->format() != "mm:ss")
    {
        setDateTimeAxisFormat("mm:ss");
    }
    else if (m_ui.timeFrameComboBox->currentData().toInt() != 60 && batteryLevelXAxis->format() != "h:mm")
    {
        setDateTimeAxisFormat("h:mm");
    }
}

void Statistics::setDateTimeAxisFormat(QString format)
{
    static_cast<QDateTimeAxis*>(m_batteryLevelChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_batteryVoltageChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_batteryCurrentChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_networkStrengthChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_uploadSpeedChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_downloadSpeedChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_cpuUsageChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_memUsageChart->axes(Qt::Horizontal)[0])->setFormat(format);
    static_cast<QDateTimeAxis*>(m_diskUsageChart->axes(Qt::Horizontal)[0])->setFormat(format);
}

void Statistics::setCurrentPage(QString page)
{
    if (page == "battery")
    {
        m_ui.menuLabel->setText("Battery");
        m_ui.batteryMenuButton->setChecked(true);
        m_ui.networkMenuButton->setChecked(false);
        m_ui.systemMenuButton->setChecked(false);

        m_ui.firstGroupBox->setTitle("Charge (%)");
        m_firstChartView->setChart(m_batteryLevelChart);
        m_ui.secondGroupBox->setTitle("Voltage (V)");
        m_secondChartView->setChart(m_batteryVoltageChart);
        m_ui.thirdGroupBox->setTitle("Current (A)");
        m_thirdChartView->setChart(m_batteryCurrentChart);

        if (m_graphTab->count() > 0)
        {
            m_graphTab->setTabText(0, "Charge");
            m_graphTab->setTabText(1, "Voltage");
            m_graphTab->setTabText(2, "Current");
        }

        m_ui.infoFrame->setVisible(false);
    }
    else if (page == "network")
    {
        m_ui.menuLabel->setText("Network");
        m_ui.batteryMenuButton->setChecked(false);
        m_ui.networkMenuButton->setChecked(true);
        m_ui.systemMenuButton->setChecked(false);

        m_ui.firstGroupBox->setTitle("Wifi strength (%)");
        m_firstChartView->setChart(m_networkStrengthChart);
        m_ui.secondGroupBox->setTitle("Download speed (Mbps)");
        m_secondChartView->setChart(m_downloadSpeedChart);
        m_ui.thirdGroupBox->setTitle("Upload speed (Mbps)");
        m_thirdChartView->setChart(m_uploadSpeedChart);

        if (m_graphTab->count() > 0)
        {
            m_graphTab->setTabText(0, "Wifi strength");
            m_graphTab->setTabText(1, "Download speed");
            m_graphTab->setTabText(2, "Upload speed");
        }

        m_ui.infoFrame->setVisible(true);
    }
    else if (page == "system")
    {
        m_ui.menuLabel->setText("System");
        m_ui.batteryMenuButton->setChecked(false);
        m_ui.networkMenuButton->setChecked(false);
        m_ui.systemMenuButton->setChecked(true);

        m_ui.firstGroupBox->setTitle("CPU (%)");
        m_firstChartView->setChart(m_cpuUsageChart);
        m_ui.secondGroupBox->setTitle("Memory (%)");
        m_secondChartView->setChart(m_memUsageChart);
        m_ui.thirdGroupBox->setTitle("Storage (%)");
        m_thirdChartView->setChart(m_diskUsageChart);

        if (m_graphTab->count() > 0)
        {
            m_graphTab->setTabText(0, "CPU");
            m_graphTab->setTabText(1, "Memory");
            m_graphTab->setTabText(2, "Storage");
        }

        m_ui.infoFrame->setVisible(false);
    }
}

float Statistics::getMaxYAxisData(QLineSeries* series, QDateTime now)
{
    QList<QPointF>::iterator i;
    QList<QPointF> list = series->points();
    float max = 0;

    int range = m_ui.timeFrameComboBox->currentData().toInt();
    QDateTime minTime = now.addSecs(-1 * range);

    for (QPointF i : list)
    {
        if (i.y() > max)
        {
            max = i.y();
        }
    }
    return max;
}
