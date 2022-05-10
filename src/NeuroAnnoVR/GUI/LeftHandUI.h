#ifndef KOUEK_LEFT_HAND_UI_H
#define KOUEK_LEFT_HAND_UI_H

#include <QtWidgets/qwidget.h>
#include <QtWidgets/qbuttongroup.h>

#include <util/QTransferFunction.h>

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
		QTransferFunctionWidget* tfWdgt;

	public:
		explicit LeftHandUI(QWidget* parent = Q_NULLPTR);
		~LeftHandUI();

		inline void setTFFromTFData(
			const std::map<uint8_t, std::array<qreal, 4>>& tfDat)
		{
			tfWdgt->loadFromTFData(tfDat);
		}
		inline const auto& getTFData() const
		{
			return tfWdgt->getTFData();
		}

	signals:
		void moveModeBtnsClicked(int id);
		void meshAlphaSliderChanged(float alpha);
		void spacesScaleChanged(float scale);
		void tfChanged();
	};
}

#endif // !KOUEK_LEFT_HAND_UI_H
