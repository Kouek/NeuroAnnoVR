#ifndef KOUEK_Q_TRANSFER_FUNCTION_H
#define KOUEK_Q_TRANSFER_FUNCTION_H

#include <array>

#include <QtGui/qpainter.h>
#include <QtWidgets/qwidget.h>
#include <QtWidgets/qpushbutton.h>
#include <QtWidgets/qboxlayout.h>

#include <QtCharts/qchart.h>
#include <QtCharts/qchartview.h>
#include <QtCharts/qscatterseries.h>
#include <QtCharts/qlineseries.h>
#include <QtCharts/qareaseries.h>

namespace kouek
{
	class QTransferFunction : public QtCharts::QChart
	{
		Q_OBJECT

	private:
		bool needUpdate = false;
		std::map<uint8_t,
			std::tuple<QPointF, std::array<qreal, 4>>> tfDat;
		// currently QScatterSeries doesn't support 
		// assigning color to individual point,
		// thus multiple QScatterSeries are needed.
		std::map<uint8_t, QtCharts::QScatterSeries*> tfPoints;
		QtCharts::QLineSeries tfLine;
		QtCharts::QAreaSeries* tfArea = nullptr;

	public:
		QTransferFunction(
			QGraphicsItem* parent = nullptr,
			Qt::WindowFlags wFlags = Qt::WindowFlags())
			: QtCharts::QChart(parent, wFlags)
		{
			setTheme(QtCharts::QChart::ChartThemeBlueCerulean);
			setTitle(QString(tr("Transfer Function")));

			tfArea = new QtCharts::QAreaSeries(&tfLine);
			QPen pen(Qt::black);
			pen.setWidth(0);
			tfArea->setPen(pen);
			addSeries(tfArea);
		}

		inline void setTFPoint(uint8_t key, const std::array<qreal, 4>& val)
		{
			QPointF tfPt((qreal)key, val[3]);
			auto itr = tfDat.find(key);
			if (itr == tfDat.end())
			{
				tfDat.emplace(std::piecewise_construct,
					std::forward_as_tuple(key),
					std::forward_as_tuple(tfPt, val));
				
				QtCharts::QScatterSeries* tfPoint = new QtCharts::QScatterSeries();
				tfPoint->setBorderColor(QColor(0, 0, 0));
				tfPoint->setMarkerShape(QtCharts::QScatterSeries::MarkerShapeCircle);
				tfPoint->setMarkerSize(15.0);
				tfPoint->append(tfPt);
				tfPoint->setColor(QColor(
					(int)(val[0] * 255),
					(int)(val[1] * 255),
					(int)(val[2] * 255)));
				tfPoints.emplace(std::piecewise_construct,
					std::forward_as_tuple(key),
					std::forward_as_tuple(tfPoint));
				addSeries(tfPoint);
			}
			else
			{
				auto& [oldTFPt, oldVal] = itr->second;
				oldTFPt = tfPt;
				oldVal = val;

				auto tfPnt = tfPoints.find(key)->second;
				tfPnt->clear();
				tfPnt->append(tfPt);
				tfPnt->setColor(QColor((int)(val[0] * 255),
					(int)(val[1] * 255),
					(int)(val[2] * 255)));
			}

			needUpdate = true;
		}

		inline void clear()
		{
			for (auto& [key, tfPnt] : tfPoints)
				tfPnt->clear();
			tfLine.clear();
		}

	protected:
		void paint(
			QPainter* painter,
			const QStyleOptionGraphicsItem* option,
			QWidget* widget = nullptr) override
		{
			if (needUpdate)
				updateSeries();
			QtCharts::QChart::paint(painter, option, widget);
		}

	private:
		inline void updateSeries()
		{
			tfLine.clear();
			for (const auto& pair : tfDat)
			{
				const auto& [tfPt, val] = pair.second;
				tfLine.append(tfPt);
			}
			tfArea->setUpperSeries(&tfLine);

			QLinearGradient gradient(QPointF(0, 0), QPointF(1.0, 0));
			for (const auto& pair : tfDat)
			{
				const auto& [tfPt, val] = pair.second;
				gradient.setColorAt(tfPt.x() / 255,
					QColor(
						(int)(val[0] * 255),
						(int)(val[1] * 255),
						(int)(val[2] * 255)));
			}
			gradient.setCoordinateMode(QGradient::ObjectMode);
			tfArea->setBrush(gradient);

			// axes shoud be updated too
			createDefaultAxes();
			axes(Qt::Horizontal).first()->setRange(0, 255);
			axes(Qt::Vertical).first()->setRange(0, 1.0);

			needUpdate = false;
		}
	};

	class QTransferFunctionView : public QWidget
	{
		Q_OBJECT

	private:
		QTransferFunction* tf;
		QtCharts::QChartView* tfView;

		QPushButton* pushButtonAddPt, * pushButtonDelPt;

	public:
		QTransferFunctionView(
			QWidget* parent = Q_NULLPTR,
			Qt::WindowFlags chartWFlags = Qt::WindowFlags())
			: QWidget(parent), tf(new QTransferFunction(nullptr, chartWFlags))
		{
			setLayout(new QVBoxLayout());

			tfView = new QtCharts::QChartView(tf);
			layout()->addWidget(tfView);

			QHBoxLayout* btnsLayout = new QHBoxLayout();
			pushButtonAddPt = new QPushButton(tr("Add Key Point"));
			pushButtonDelPt = new QPushButton(tr("Delete Key Point"));
			btnsLayout->addWidget(pushButtonAddPt);
			btnsLayout->addWidget(pushButtonDelPt);
			layout()->addItem(btnsLayout);
		}

		inline void loadFromTFData(
			const std::map<uint8_t, std::array<qreal, 4>>& tfDat)
		{
			tf->clear();
			for (const auto& [key, val] : tfDat)
				tf->setTFPoint(key, val);
		}
	};
}

#endif // !KOUEK_Q_TRANSFER_FUNCTION_H
