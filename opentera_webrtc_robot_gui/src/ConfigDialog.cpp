#include "ConfigDialog.h"

#include "ui_ConfigDialog.h"

ConfigDialog::ConfigDialog(QWidget *parent)
  :  m_ui(new Ui::ConfigDialog())
{
    m_ui->setupUi(this);

}

ConfigDialog::~ConfigDialog()
{

}