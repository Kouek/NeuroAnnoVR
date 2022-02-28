#include <QtWidgets/qlayout.h>

#include "DemoWindow.h"
#include "ui_demowindow.h"

kouek::DemoWindow::DemoWindow(QWidget* parent)
	: QWidget(parent),
	ui(new Ui::DemoWindow),
	volumeView(new VolumeView)
{
	ui->setupUi(this);

	ui->groupBoxView->setLayout(new QVBoxLayout);
	ui->groupBoxView->layout()->addWidget(volumeView);
}

kouek::DemoWindow::~DemoWindow()
{
	delete ui;
}
