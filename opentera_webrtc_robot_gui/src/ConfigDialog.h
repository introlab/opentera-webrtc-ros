#ifndef _CONFIG_DIALOG_H_
#define _CONFIG_DIALOG_H_

#include <QDialog>

QT_BEGIN_NAMESPACE
    namespace Ui
    {
        class ConfigDialog;
    }
QT_END_NAMESPACE

class MainWindow;

class ConfigDialog : public QDialog
{
    Q_OBJECT

public:
    ConfigDialog(MainWindow* parent = nullptr);
    ~ConfigDialog();

    int getMicVolumeSliderValue();
    void setMicVolumeSliderValue(int value);
    int getVolumeSliderValue();
    void setVolumeSliderValue(int value);

protected:
    Ui::ConfigDialog* m_ui;
};

#endif  //#define _CONFIG_DIALOG_H_
