#ifndef KOUEK_LEFT_HAND_UI_H
#define KOUEK_LEFT_HAND_UI_H

#include <QtWidgets/qwidget.h>
#include <QtWidgets/qbuttongroup.h>

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

		QButtonGroup* moveModes;

	public:
		explicit LeftHandUI(QWidget* parent = Q_NULLPTR);
		~LeftHandUI();

	signals:
		void moveModeBtnsClicked(int id);
		void meshAlphaSliderChanged(float alpha);
	};
}

#endif // !KOUEK_LEFT_HAND_UI_H
