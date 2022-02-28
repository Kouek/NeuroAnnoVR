#ifndef KOUEK_DEMO_WINDOW_H
#define KOUEK_DEMO_WINDOW_H

#include <QtWidgets/qwidget.h>

#include "VolumeView.h"

namespace Ui
{
	class DemoWindow;
}

namespace kouek
{
	class DemoWindow : public QWidget
	{
		Q_OBJECT

	public:
		explicit DemoWindow(QWidget* parent = Q_NULLPTR);
		~DemoWindow();

	private:
		Ui::DemoWindow* ui;

		VolumeView* volumeView;
	};
}

#endif // !KOUEK_DEMO_WINDOW_H
