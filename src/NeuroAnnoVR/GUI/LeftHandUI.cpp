#include "LeftHandUI.h"
#include "ui_lefthandui.h"

kouek::LeftHandUI::LeftHandUI(QWidget* parent)
	: QWidget(parent), ui(new Ui::LeftHandUI)
{
	ui->setupUi(this);

	tfWdgt = new QTransferFunctionWidget();
	tfWdgt->setStyleSheet(
		"font: 20pt \"Times New Roman\";");
	ui->groupBoxTF->setLayout(new QVBoxLayout());
	ui->groupBoxTF->layout()->addWidget(tfWdgt);

	moveModes = new QButtonGroup(this);
	moveModes->addButton(ui->pushButtonWander, 0);
	moveModes->addButton(ui->pushButtonFocus, 1);
	connect(moveModes, static_cast<void(QButtonGroup::*)(int)>(
		&QButtonGroup::buttonClicked), [&](int id) {
			emit moveModeBtnsClicked(id);
		});
	connect(ui->verticalSliderMeshAlpha, &QSlider::valueChanged,
		[&](int val) {
			double alpha = (double)val / (double)ui->verticalSliderMeshAlpha->maximum();
			ui->labelMeshAlpha->setText(QString("%1").arg(alpha, 0, 'f', 2));
			emit meshAlphaSliderChanged(alpha);
		});
	connect(tfWdgt, &QTransferFunctionWidget::tfChanged,
		this, &LeftHandUI::tfChanged);
}

kouek::LeftHandUI::~LeftHandUI()
{
}
