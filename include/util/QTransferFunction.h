#ifndef KOUEK_Q_TRANSFER_FUNCTION_H
#define KOUEK_Q_TRANSFER_FUNCTION_H

#include <array>

#include <QtGui/qpainter.h>
#include <QtWidgets/qwidget.h>
#include <QtWidgets/qlabel.h>
#include <QtWidgets/qcolordialog.h>
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
		bool needUpdate = true;
		std::map<uint8_t,
			std::tuple<QPointF, std::array<qreal, 4>>> tfDat;
		// currently QScatterSeries doesn't support 
		// assigning color to individual point,
		// thus multiple QScatterSeries are needed.
		std::map<uint8_t, QtCharts::QScatterSeries*> tfPoints;
		QtCharts::QScatterSeries* intrctPoint;

		QPen borderPen;
		QtCharts::QLineSeries* tfLine;
		QtCharts::QAreaSeries* tfArea = nullptr;

	public:
		QTransferFunction(
			QGraphicsItem* parent = nullptr,
			Qt::WindowFlags wFlags = Qt::WindowFlags())
			: QtCharts::QChart(parent, wFlags), borderPen(Qt::white)
		{
			borderPen.setWidth(2);

			setTheme(QtCharts::QChart::ChartThemeBlueCerulean);
			setTitle(QString(tr("Transfer Function")));
			legend()->hide();

			tfLine = new QtCharts::QLineSeries();
			tfLine->setColor(Qt::white);
			tfLine->setPen(borderPen);
			addSeries(tfLine);

			QPen zeroPen;
			zeroPen.setWidth(0);
			tfArea = new QtCharts::QAreaSeries(tfLine);
			tfArea->setPen(zeroPen);
			addSeries(tfArea);

			intrctPoint = new QtCharts::QScatterSeries();
			intrctPoint->setColor(Qt::black);
			intrctPoint->setPen(borderPen);
			intrctPoint->setMarkerShape(QtCharts::QScatterSeries::MarkerShapeRectangle);
			intrctPoint->setMarkerSize(10.0);
			addSeries(intrctPoint);
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
				
				QtCharts::QScatterSeries* tfPoint = nullptr;
				if (auto itr = tfPoints.find(key); itr != tfPoints.end())
					tfPoint = itr->second;
				else
				{
					tfPoint = new QtCharts::QScatterSeries();
					tfPoints.emplace(std::piecewise_construct,
						std::forward_as_tuple(key),
						std::forward_as_tuple(tfPoint));
					addSeries(tfPoint);
				}
				tfPoint->setPen(borderPen);
				tfPoint->setMarkerShape(QtCharts::QScatterSeries::MarkerShapeCircle);
				tfPoint->setMarkerSize(15.0);
				tfPoint->append(tfPt);
				tfPoint->setColor(QColor(
					(int)(val[0] * 255),
					(int)(val[1] * 255),
					(int)(val[2] * 255)));
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
		inline void deleteTFPoint(uint8_t key)
		{
			tfPoints.at(key)->clear();
			tfDat.erase(key);
			needUpdate = true;
		}
		inline void clear()
		{
			for (auto& [key, tfPnt] : tfPoints)
				tfPnt->clear();
			tfLine->clear();
		}
		inline const auto& getTFData() const
		{
			return tfDat;
		}
		inline const auto getInteractionPoint() const
		{
			return intrctPoint->points().first();
		}
		inline void unsetInteractionPoint()
		{
			intrctPoint->clear();
		}
		inline void setInteractionPoint(const QPointF& newPt)
		{
			intrctPoint->clear();
			intrctPoint->append(newPt);
		}
		inline void setTFPointFromInteractionPoint(const QColor& color)
		{
			const auto pt = intrctPoint->points().first();
			uint8_t key = pt.x();
			std::array<qreal, 4> val{ color.redF(), color.greenF(),
				color.blueF(), pt.y() };
			setTFPoint(key, val);
		}
		inline uint8_t resetTFPoint(uint8_t key, const QPointF& newPt)
		{
			uint8_t newKey = newPt.x();
			auto [pt, newVal] = tfDat.at(key);
			newVal[3] = newPt.y();
			if (newKey != key)
				deleteTFPoint(key);
			setTFPoint(newKey, newVal);
			return newKey;
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
			tfLine->clear();
			for (const auto& pair : tfDat)
			{
				const auto& [tfPt, val] = pair.second;
				tfLine->append(tfPt);
			}
			tfArea->setUpperSeries(tfLine);

			QLinearGradient gradient(QPointF(0, 0), QPointF(1.0, 0));
			qreal length = 255.0;
			qreal start = 0;
			if (auto [beg, rbeg] = std::tuple{ tfDat.begin(),tfDat.rbegin() };
				beg != tfDat.end() && rbeg != tfDat.rend())
			{
				start = std::get<QPointF>(beg->second).x();
				if (beg->first != rbeg->first)
					length = std::get<QPointF>(rbeg->second).x()
					- std::get<QPointF>(beg->second).x();
			}
			for (const auto& pair : tfDat)
			{
				const auto& [tfPt, val] = pair.second;
				gradient.setColorAt(
					(tfPt.x() -start) / length,
					QColor(
						(int)(val[0] * 255),
						(int)(val[1] * 255),
						(int)(val[2] * 255)));
			}
			gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
			tfArea->setBrush(gradient);

			// axes shoud be updated too
			createDefaultAxes();
			axes(Qt::Horizontal).first()->setRange(0, 255);
			axes(Qt::Vertical).first()->setRange(0, 1.0);

			needUpdate = false;
		}
	};

	class QTransferFunctionView : public QtCharts::QChartView
	{
		Q_OBJECT

	private:
		bool isTFChanged = false;
		int exstIntrctKey = -1;

	public:
		QTransferFunctionView(
			QTransferFunction* tf,
			QWidget* parent = Q_NULLPTR)
			: QtCharts::QChartView(tf, parent)
		{}
		inline bool hasExstIntrctPnt() const
		{
			return exstIntrctKey != -1;
		}
		inline void setInteractionPointColor(const QColor& color)
		{
			auto tf = static_cast<QTransferFunction*>(chart());
			if (hasExstIntrctPnt())
			{
				auto newPair = tf->getTFData().at(exstIntrctKey);
				auto& [pt, val] = newPair;
				val[0] = color.redF();
				val[1] = color.greenF();
				val[2] = color.blueF();
				tf->setTFPoint(exstIntrctKey, val);
			}
			else
				tf->setTFPointFromInteractionPoint(color);
		}
		inline void deleteExstIntrctPnt()
		{
			auto tf = static_cast<QTransferFunction*>(chart());
			tf->deleteTFPoint(exstIntrctKey);
		}

	signals:
		void pntInfoChnged(const QString& info);
		void tfChanged();

	protected:
		void mousePressEvent(QMouseEvent* e) override
		{
			constexpr qreal ERR = 100.0;

			exstIntrctKey = -1;
			qreal minDiffSqr = std::numeric_limits<qreal>::max();
			auto tf = static_cast<QTransferFunction*>(chart());
			for (const auto& [key, pair] : tf->getTFData())
			{
				const auto& [pt, val] = pair;
				QPointF dlt = tf->mapToPosition(pt) - e->pos();
				qreal diffSqr = QPointF::dotProduct(dlt, dlt);
				if (diffSqr <= ERR && diffSqr < minDiffSqr)
				{
					exstIntrctKey = key;
					minDiffSqr = diffSqr;
				}
			}

			if (hasExstIntrctPnt())
				tf->unsetInteractionPoint();
			else
			{
				auto newPt = tf->mapToValue(e->pos());
				if (newPt.x() < 0) newPt.setX(0);
				if (newPt.x() >= 255.0) newPt.setX(255.f);
				if (newPt.y() < 0)newPt.setY(0);
				if (newPt.y() >= 1.0)newPt.setY(1.f);
				tf->setInteractionPoint(newPt);
			}
		}
		void mouseMoveEvent(QMouseEvent* e) override
		{
			auto tf = static_cast<QTransferFunction*>(chart());
			if ((e->buttons() & Qt::LeftButton) != 0
				&& hasExstIntrctPnt())
			{
				auto newPt = tf->mapToValue(e->pos());
				if (newPt.x() < 0) newPt.setX(0);
				if (newPt.x() >= 255.0) newPt.setX(255.f);
				if (newPt.y() < 0)newPt.setY(0);
				if (newPt.y() >= 1.0)newPt.setY(1.f);
				auto newKey = tf->resetTFPoint(exstIntrctKey, newPt);
				exstIntrctKey = newKey;
				const auto& [pt, val] = tf->getTFData().at(exstIntrctKey);
				isTFChanged = true;
				emit pntInfoChnged(QString("%0 -> (%1, %2, %3, %4)")
					.arg(exstIntrctKey)
					.arg(val[0]).arg(val[1]).arg(val[2]).arg(val[3]));
			}
		}
		void mouseReleaseEvent(QMouseEvent* e) override
		{
			auto tf = static_cast<QTransferFunction*>(chart());
			if (hasExstIntrctPnt())
			{
				const auto& [pt, val] = tf->getTFData().at(exstIntrctKey);
				emit pntInfoChnged(QString("%0 -> (%1, %2, %3, %4)")
					.arg(exstIntrctKey)
					.arg(val[0]).arg(val[1]).arg(val[2]).arg(val[3]));
			}
			else
			{
				const auto& pt = tf->getInteractionPoint();
				emit pntInfoChnged(QString("(%0, %1)")
					.arg(pt.x()).arg(pt.y()));
			}

			if (isTFChanged)
			{
				emit tfChanged();
				isTFChanged = false;
			}
		}
	};

	class QTransferFunctionWidget : public QWidget
	{
		Q_OBJECT

	private:
		QTransferFunction* tf;

		QLabel* pntInfo;
		QTransferFunctionView* tfView;
		QColorDialog* colorDiag;

		QPushButton* pushButtonAddPt;
		QPushButton* pushButtonChngColor;
		QPushButton* pushButtonDelPt;

		std::vector<QPushButton*> btnList;

	public:
		QTransferFunctionWidget(
			QWidget* parent = Q_NULLPTR,
			Qt::WindowFlags chartWFlags = Qt::WindowFlags())
			: QWidget(parent), tf(new QTransferFunction(nullptr, chartWFlags))
		{
			pntInfo = new QLabel();
			tfView = new QTransferFunctionView(tf);
			colorDiag = new QColorDialog();

			setLayout(new QVBoxLayout());
			layout()->addWidget(pntInfo);
			layout()->addWidget(tfView);
			QHBoxLayout* colorLayout = new QHBoxLayout();
			colorLayout->addStretch();
			colorLayout->addWidget(colorDiag);
			colorLayout->addStretch();
			colorDiag->hide();
			static_cast<QVBoxLayout*>(layout())->addLayout(colorLayout);

			pushButtonAddPt = new QPushButton(tr("Add Key Point"));
			pushButtonChngColor = new QPushButton(tr("Change Key Point Color"));
			pushButtonDelPt = new QPushButton(tr("Delete Key Point"));
			pushButtonDelPt->setStyleSheet(
				"color: red;");
			btnList.emplace_back(pushButtonAddPt);
			btnList.emplace_back(pushButtonChngColor);
			btnList.emplace_back(pushButtonDelPt);

			QHBoxLayout* btnsLayout = new QHBoxLayout();
			btnsLayout->addWidget(pushButtonAddPt);
			btnsLayout->addWidget(pushButtonChngColor);
			btnsLayout->addWidget(pushButtonDelPt);
			static_cast<QVBoxLayout*>(layout())->addLayout(btnsLayout);

			connect(tfView, &QTransferFunctionView::pntInfoChnged,
				[&](const QString& info) {
					pntInfo->setText(info);
				});
			connect(pushButtonAddPt, &QPushButton::clicked,
				[&]() {
					if (tfView->hasExstIntrctPnt()) return;
					pntInfo->hide();
					tfView->hide();
					colorDiag->show();
					for (auto btn : btnList)
						btn->hide();
				});
			connect(pushButtonChngColor, &QPushButton::clicked,
				[&]() {
					if (!tfView->hasExstIntrctPnt()) return;
					pntInfo->hide();
					tfView->hide();
					colorDiag->show();
					for (auto btn : btnList)
						btn->hide();
				});
			connect(pushButtonDelPt, &QPushButton::clicked,
				[&]() {
					if (!tfView->hasExstIntrctPnt()) return;
					tfView->deleteExstIntrctPnt();
					emit tfChanged();
				});
			connect(colorDiag, &QColorDialog::rejected,
				[&]() {
					pntInfo->show();
					tfView->show();
					colorDiag->hide();
					for (auto btn : btnList)
						btn->show();
				});
			connect(colorDiag, &QColorDialog::colorSelected,
				[&](const QColor& color) {
					tfView->setInteractionPointColor(color);
					pntInfo->show();
					tfView->show();
					colorDiag->hide();
					for (auto btn : btnList)
						btn->show();
					emit tfChanged();
				});

			connect(tfView, &QTransferFunctionView::tfChanged,
				this, &QTransferFunctionWidget::tfChanged);
		}
		inline void loadFromTFData(
			const std::map<uint8_t, std::array<qreal, 4>>& tfDat)
		{
			tf->clear();
			for (const auto& [key, val] : tfDat)
				tf->setTFPoint(key, val);
		}
		inline const auto& getTFData() const
		{
			return tf->getTFData();
		}

	signals:
		void tfChanged();
	};
}

#endif // !KOUEK_Q_TRANSFER_FUNCTION_H
