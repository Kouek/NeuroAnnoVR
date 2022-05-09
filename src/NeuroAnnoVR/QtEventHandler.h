#ifndef KOUEK_QT_EVENT_HANDLER_H
#define KOUEK_QT_EVENT_HANDLER_H

#include "EventHandler.h"
#include "GUI/EditorWindow.h"
#include "GUI/LeftHandUI.h"
#include "GUI/RightHandUI.h"

#include <QtCore/qtimer.h>
#include <QtGui/qopenglcontext.h>
#include <QtGui/qopenglframebufferobject.h>
#include <QtGui/qopenglpaintdevice.h>
#include <QtGui/qpainter.h>
#include <QtWidgets/qgraphicsscene.h>
#include <QtWidgets/qgraphicsitem.h>

#include <CMakeIn.h>
#include <util/VolumeCfg.h>

namespace kouek
{
	struct HandUIHandler : public QObject
	{
		EditorWindow* glCtxProvider;
		AppStates* states;
		
		std::array<QWidget*, 2> wdgt2;
		std::array<QGraphicsScene*, 2> scn2;
		std::array<QOpenGLFramebufferObject*, 2> FBO2;
		std::array<QOpenGLPaintDevice*, 2> glDvc2;
		std::array<QPainter*, 2> pntr2;

		QTimer* timer;

		QPointF lastMousePos;
		Qt::MouseButtons lastMouseButtons;

		HandUIHandler(EditorWindow* glCtxProvider, AppStates* states);
		~HandUIHandler();

		void onTimeOut();
		void onLeftHandSceneChanged(const QList<QRectF>& region);
		void onRightHandSceneChanged(const QList<QRectF>& region);
	};

	class QtEventHandler : public EventHandler
	{
	private:
		bool isTFChanged = false;
		bool lastLeftHandUIShown = false;
		std::array<float, 3> moveSteps = { 0 };
		std::array<float, 3> subrgnMoveSteps = { 0 };

		HandUIHandler handUI;

	public:
		QtEventHandler(
			EditorWindow* sender,
			std::shared_ptr<AppStates> sharedStates);
		void update() override;
		inline GLuint getHandUITex(uint8_t handIdx) const
		{
			return handUI.FBO2[handIdx]->texture();
		}
		inline auto getHandUISize(uint8_t hndIdx) const
		{
			return handUI.wdgt2[hndIdx]->size();
		}
	};
}

#endif // !KOUEK_QT_EVENT_HANDLER_H
