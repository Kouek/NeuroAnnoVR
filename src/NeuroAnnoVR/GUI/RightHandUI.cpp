#include "RightHandUI.h"
#include "ui_righthandui.h"

kouek::RightHandUI::RightHandUI(QWidget* parent)
	: QWidget(parent), ui(new Ui::RightHandUI)
{
	ui->setupUi(this);

	intrctActModes = new QButtonGroup(this);
	intrctActModes->addButton(ui->pushButtonSelectVertex, 0);
	intrctActModes->addButton(ui->pushButtonAddPath, 1);
	intrctActModes->addButton(ui->pushButtonAddVertex, 2);
	intrctActModes->addButton(ui->pushButtonDeleteVertex, 3);
	intrctActModes->addButton(ui->pushButtonJoinPath, 4);
	connect(intrctActModes, static_cast<void(QButtonGroup::*)(int)>(
		&QButtonGroup::buttonClicked), [&](int id) {
			emit interactionActionModeBtnsClicked(id);
		});

	intrctModes = new QButtonGroup(this);
	intrctModes->addButton(ui->pushButtonAnnoBall, 0);
	intrctModes->addButton(ui->pushButtonAnnoLaser, 1);
	connect(intrctModes, static_cast<void(QButtonGroup::*)(int)>(
		&QButtonGroup::buttonClicked), [&](int id) {
			emit interactionModeBtnsClicked(id);
		});
}

kouek::RightHandUI::~RightHandUI()
{
}
