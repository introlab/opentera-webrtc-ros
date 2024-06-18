#include "MainWindow.h"

#include <QApplication>
#include <QImageReader>
#include <QImage>
#include <QDebug>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <QThread>
#include <QFile>
#include <initializer_list>
#include <signal.h>
#include <unistd.h>

void catchUnixSignals(std::initializer_list<int> quitSignals)
{
    auto handler = [](int /* sig */) -> void { QCoreApplication::quit(); };

    sigset_t blockingMask;
    sigemptyset(&blockingMask);
    for (auto sig : quitSignals)
    {
        sigaddset(&blockingMask, sig);
    }

    struct sigaction sa;
    sa.sa_handler = handler;
    sa.sa_mask = blockingMask;
    sa.sa_flags = 0;

    for (auto sig : quitSignals)
    {
        sigaction(sig, &sa, nullptr);
    }
}

class AsyncSpinner
{
public:
    template<typename... Nodes>
    void start_spin(std::shared_ptr<Nodes>... nodes)
    {
        spinner = std::thread([nodes...] { rclcpp::spin(nodes...); });
    }

    void stop()
    {
        rclcpp::shutdown();
        spinner.join();
    }

private:
    std::thread spinner;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto nh = std::make_shared<rclcpp::Node>("opentera_webrtc_robot_gui_node");
    bool fullScreen = nh->declare_parameter("fullScreen", false);

    nh->declare_parameter("device_properties_path", rclcpp::PARAMETER_STRING);
    std::string jsonFilePath;
    if (!nh->get_parameter("device_properties_path", jsonFilePath))
    {
        RCLCPP_ERROR(nh->get_logger(), "The parameter device_properties_path is required.");
        return 1;
    }

    AsyncSpinner spinner;

    catchUnixSignals({SIGQUIT, SIGINT, SIGTERM, SIGHUP});

    // Hides an internal error that comes from resizing a QGLWidget, which doesn't affect our use
    qputenv("QT_LOGGING_RULES", QByteArray("*.debug=false;qt.qpa.xcb=false"));

    QApplication app(argc, argv);
    // Stylesheet
    QFile file(":/stylesheet.qss");
    file.open(QFile::ReadOnly);
    QString stylesheet = QLatin1String(file.readAll());
    app.setStyleSheet(stylesheet);


    MainWindow w(QString::fromStdString(jsonFilePath), *nh);
    if (fullScreen)
    {
        w.showFullScreen();
    }
    else
    {
        w.show();
    }

    // Load test image from QRC
    QImage testImage(":/Test_640x480.png");

    qDebug() << testImage;
    w.setImage(testImage);

    // Will start ROS loop in background
    spinner.start_spin(nh);

    // will run app event loop (infinite)
    // qDebug() << "MainThread " << QThread::currentThread();
    app.exec();

    // Stop ROS loop
    spinner.stop();

    w.endCall();

    // Fixes a bug where the program would stay open because of the camera window
    w.closeCameraWindow();
}
