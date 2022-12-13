#include "ConfigDialog.h"
#include "MainWindow.h"

ConfigDialog::ConfigDialog(MainWindow* parent)
{
    m_ui.setupUi(this);
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint | Qt::WindowTitleHint);

    // Sliders
    m_ui.micVolumeSlider->setValue(100);
    connect(m_ui.micVolumeSlider, &QSlider::valueChanged, parent, &MainWindow::onMicVolumeSliderValueChanged);
    m_ui.volumeSlider->setValue(100);
    connect(m_ui.volumeSlider, &QSlider::valueChanged, parent, &MainWindow::onVolumeSliderValueChanged);
    m_ui.opacitySlider->setValue(parent->m_deviceProperties.defaultLocalCameraOpacity);
    connect(m_ui.opacitySlider, &QSlider::valueChanged, parent, &MainWindow::onOpacitySliderValueChanged);
}

int ConfigDialog::getMicVolumeSliderValue()
{
    return m_ui.micVolumeSlider->value();
}

void ConfigDialog::setMicVolumeSliderValue(int value)
{
    m_ui.micVolumeSlider->setValue(value);
}

int ConfigDialog::getVolumeSliderValue()
{
    return m_ui.volumeSlider->value();
}

void ConfigDialog::setVolumeSliderValue(int value)
{
    m_ui.volumeSlider->setValue(value);
}

int ConfigDialog::getOpacitySliderValue()
{
    return m_ui.opacitySlider->value();
}

void ConfigDialog::setOpacitySliderValue(int value)
{
    if (value == 0)
    {
        m_lastOpacityValue = m_ui.opacitySlider->value();
    }
    if (m_ui.opacitySlider->value() == 0)
    {
        value = m_lastOpacityValue;
    }
    m_ui.opacitySlider->setValue(value);
}
