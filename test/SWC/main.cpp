#include <array>

#include "GLView.h"

using namespace kouek;

int main(int argc, char** argv)
{
	QApplication app(argc, argv);
	GLView view;
	view.show();
	return app.exec();

	return 0;
}
