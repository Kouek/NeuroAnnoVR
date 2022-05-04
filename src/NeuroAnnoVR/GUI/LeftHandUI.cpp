#include "LeftHandUI.h"
#include "ui_lefthandui.h"

kouek::LeftHandUI::LeftHandUI(QWidget* parent)
	: QWidget(parent), ui(new Ui::LeftHandUI)
{
	ui->setupUi(this);

	moveModes = new QButtonGroup(this);
	moveModes->addButton(ui->pushButtonWander, 0);
	moveModes->addButton(ui->pushButtonFocus, 1);
	connect(moveModes, static_cast<void(QButtonGroup::*)(int)>(
		&QButtonGroup::buttonClicked), [&](int id) {
			emit moveModeBtnsClicked(id);
		});
	connect(ui->horizontalSliderMeshAlpha, &QSlider::valueChanged,
		[&](int val) {
			double alpha = (double)val / (double)ui->horizontalSliderMeshAlpha->maximum();
			ui->labelMeshAlpha->setText(QString("%1").arg(alpha, 0, 'f', 2));
			emit meshAlphaSliderChanged(alpha);
		});
}

kouek::LeftHandUI::~LeftHandUI()
{
}
