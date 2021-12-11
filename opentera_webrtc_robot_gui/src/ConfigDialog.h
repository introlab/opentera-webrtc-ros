#ifndef _CONFIG_DIALOG_H_
#define _CONFIG_DIALOG_H_

#include <QDialog>


QT_BEGIN_NAMESPACE
namespace Ui { class ConfigDialog; }
QT_END_NAMESPACE


class ConfigDialog : public QDialog
{
    Q_OBJECT

    public:
    ConfigDialog(QWidget *parent=nullptr);
    ~ConfigDialog();

    protected:

    Ui::ConfigDialog *m_ui;
};

#endif //#define _CONFIG_DIALOG_H_
