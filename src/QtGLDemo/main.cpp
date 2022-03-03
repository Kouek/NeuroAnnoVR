#include <QtWidgets/qapplication.h>

#include "DemoWindow.h"

int main(int argc, char** argv)
{
	QApplication app(argc, argv);
	kouek::DemoWindow window;
	window.showMaximized();
	return app.exec();
}
