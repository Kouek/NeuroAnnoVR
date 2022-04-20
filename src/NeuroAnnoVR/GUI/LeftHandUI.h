#ifndef KOUEK_LEFT_HAND_UI_H
#define KOUEK_LEFT_HAND_UI_H

#include <QtWidgets/qwidget.h>

namespace Ui
{
	class LeftHandUI;
}

namespace kouek
{
	class LeftHandUI : public QWidget
	{
		Q_OBJECT

	private:
		Ui::LeftHandUI* ui;

	public:
		explicit LeftHandUI(QWidget* parent = Q_NULLPTR);
		~LeftHandUI();
	};
}

#endif // !KOUEK_LEFT_HAND_UI_H
