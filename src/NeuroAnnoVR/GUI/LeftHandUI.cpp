#include "LeftHandUI.h"
#include "ui_lefthandui.h"

kouek::LeftHandUI::LeftHandUI(QWidget* parent)
	: QWidget(parent), ui(new Ui::LeftHandUI)
{
	ui->setupUi(this);
}

kouek::LeftHandUI::~LeftHandUI()
{
}
