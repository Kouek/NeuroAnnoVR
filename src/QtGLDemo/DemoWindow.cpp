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

	constexpr std::array<float, 3> SCALES = { 5.f,5.f,5.f };
	connect(ui->horizontalSliderHalfW, &QSlider::valueChanged,
		[&, SCALES](int val) {
			float size = SCALES[0] * val / ui->horizontalSliderHalfW->maximum();
			ui->labelHalfWNum->setText(QString::number(size, 'f', 2));
			volumeView->setSubregionHalfW(size);
		});
	connect(ui->horizontalSliderHalfH, &QSlider::valueChanged,
		[&, SCALES](int val) {
			float size = SCALES[1] * val / ui->horizontalSliderHalfH->maximum();
			ui->labelHalfHNum->setText(QString::number(size, 'f', 2));
			volumeView->setSubregionHalfH(size);
		});
	connect(ui->horizontalSliderHalfD, &QSlider::valueChanged,
		[&, SCALES](int val) {
			float size = SCALES[2] * val / ui->horizontalSliderHalfD->maximum();;
			ui->labelHalfDNum->setText(QString::number(size, 'f', 2));
			volumeView->setSubregionHalfD(size);
		});
	connect(ui->horizontalSliderRotateY, &QSlider::valueChanged,
		[&](int val) {
			float deg = val;
			ui->labelRotateYNum->setText(QString::number(deg));
			volumeView->setSubregionRotationY(deg);
		});
	
	// sync default val from ui to deeper logic
	ui->horizontalSliderHalfW->valueChanged(ui->horizontalSliderHalfW->value());
	ui->horizontalSliderHalfH->valueChanged(ui->horizontalSliderHalfH->value());
	ui->horizontalSliderHalfD->valueChanged(ui->horizontalSliderHalfD->value());
	ui->horizontalSliderRotateY->valueChanged(ui->horizontalSliderRotateY->value());
}

kouek::DemoWindow::~DemoWindow()
{
	delete ui;
}
