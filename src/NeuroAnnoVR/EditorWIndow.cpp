#include "EditorWindow.h"
#include "ui_editorwindow.h"

#include <QtWidgets/qlayout.h>

kouek::EditorWindow::EditorWindow(QWidget* parent)
	: QWidget(parent), ui(new Ui::EditorWindow)
{
	ui->setupUi(this);
	{
		QSurfaceFormat surfaceFmt;
		surfaceFmt.setDepthBufferSize(24);
		surfaceFmt.setStencilBufferSize(8);
		surfaceFmt.setVersion(4, 5);
		surfaceFmt.setProfile(QSurfaceFormat::CoreProfile);
		
		vrView = new VRView;
		vrView->setFormat(surfaceFmt);
		ui->groupBoxVRView->layout()->addWidget(vrView);
	}
}

kouek::EditorWindow::~EditorWindow()
{
}

void kouek::EditorWindow::closeEvent(QCloseEvent* e)
{
	emit closed();
	QWidget::closeEvent(e);
}
