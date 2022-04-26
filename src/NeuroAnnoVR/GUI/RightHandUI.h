#ifndef KOUEK_RIGHT_HAND_UI_H
#define KOUEK_RIGHT_HAND_UI_H

#include <QtWidgets/qwidget.h>
#include <QtWidgets/qbuttongroup.h>

namespace Ui
{
	class RightHandUI;
}

namespace kouek
{
	class RightHandUI : public QWidget
	{
		Q_OBJECT

	private:
		Ui::RightHandUI* ui;
		
		QButtonGroup* intrctActModes;

	public:
		explicit RightHandUI(QWidget* parent = Q_NULLPTR);
		~RightHandUI();

	signals:
		void interactionActionModeBtnsClicked(int id);
	};
}

#endif // !KOUEK_RIGHT_HAND_UI_H
