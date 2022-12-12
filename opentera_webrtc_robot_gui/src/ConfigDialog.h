#ifndef _CONFIG_DIALOG_H_
#define _CONFIG_DIALOG_H_

#include <QDialog>
#include "ui_ConfigDialog.h"


class MainWindow;

class ConfigDialog : public QDialog
{
    Q_OBJECT

public:
    ConfigDialog(MainWindow* parent = nullptr);
    ~ConfigDialog() = default;

    int getMicVolumeSliderValue();
    void setMicVolumeSliderValue(int value);
    int getVolumeSliderValue();
    void setVolumeSliderValue(int value);
    int getOpacitySliderValue();
    void setOpacitySliderValue(int value);

protected:
    Ui::ConfigDialog m_ui;

private:
    int m_lastOpacityValue;
};

#endif  //#define _CONFIG_DIALOG_H_
