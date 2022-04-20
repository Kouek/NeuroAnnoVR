#ifndef KOUEK_QT_EVENT_HANDLER_H
#define KOUEK_QT_EVENT_HANDLER_H

#include "EventHandler.h"
#include "GUI/EditorWindow.h"
#include "GUI/LeftHandUI.h"

#include <QtCore/qtimer.h>
#include <QtGui/qopenglcontext.h>
#include <QtGui/qopenglframebufferobject.h>
#include <QtGui/qoffscreensurface.h>
#include <QtWidgets/qgraphicsscene.h>

#include <CMakeIn.h>
#include <util/VolumeCfg.h>

namespace kouek
{
	struct HandUIHandler : public QObject
	{
		inline static constexpr std::array<std::string_view, 2> OVERLAY_KEY_2 = {
			"kouek.leftHandUI", "kouek.rightHandUI" };
		inline static constexpr std::array<std::string_view, 2> OVERLAY_NAME_2 = {
			"Left Hand UI", "Right Hand UI" };

		bool canRun = true;

		std::array<QWidget*, 2> wdgt2;
		std::array<QGraphicsScene*, 2> scn2;
		std::array<QOffscreenSurface*, 2> offScrnSurf2;
		std::array<QOpenGLContext*, 2> ctx2;
		std::array<QOpenGLFramebufferObject*, 2> FBO2;

		std::array<vr::VROverlayHandle_t, 2> overlayHandle2;
		std::array<vr::VROverlayHandle_t, 2> overlayThumbnailHandle2;

		QTimer* timer;

		QFlags<Qt::MouseButton> lastMouseBtn;
		QPointF lastMouse;

		HandUIHandler();
		~HandUIHandler();

		void onTimeOut();
		void onLeftHandSceneChanged(const QList<QRectF>& region);
	};

	class QtEventHandler : public EventHandler
	{
	private:
		std::array<float, 3> moveSteps = { 0 };
		std::array<float, 3> subrgnMoveSteps = { 0 };

		HandUIHandler handUI;

	public:
		QtEventHandler(
			EditorWindow* sender,
			std::shared_ptr<AppStates> sharedStates);
		void update() override;
		void updateWhenDrawingOverlay();
		void updateWhenDrawingCompositor();
	};
}

#endif // !KOUEK_QT_EVENT_HANDLER_H
