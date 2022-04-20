#include "QtEventHandler.h"

#include <QtGui/qopenglpaintdevice.h>
#include <QtGui/qpainter.h>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qgraphicssceneevent.h>

kouek::HandUIHandler::HandUIHandler()
{
	QSurfaceFormat surfFmt;
	surfFmt.setDepthBufferSize(24);
	surfFmt.setStencilBufferSize(8);
	surfFmt.setVersion(4, 5);
	surfFmt.setProfile(QSurfaceFormat::CompatibilityProfile);

	VRContext::forHandsDo([&](uint8_t handIdx) {
		ctx2[handIdx] = new QOpenGLContext();
		ctx2[handIdx]->setFormat(surfFmt);
		auto ok = ctx2[handIdx]->create();
		if (!ok)
		{
			canRun = false;
			return;
		}
		offScrnSurf2[handIdx] = new QOffscreenSurface();
		offScrnSurf2[handIdx]->create();
		ctx2[handIdx]->makeCurrent(offScrnSurf2[handIdx]);

		scn2[handIdx] = new QGraphicsScene();
		});
	if (!canRun) return;

	QObject::connect(scn2[0], &QGraphicsScene::changed,
		this, &HandUIHandler::onLeftHandSceneChanged);

	wdgt2[VRContext::Hand_Left] = new LeftHandUI();
	wdgt2[VRContext::Hand_Left]->move(0, 0);
	scn2[VRContext::Hand_Left]->addWidget(wdgt2[VRContext::Hand_Left]);
	FBO2[VRContext::Hand_Left] = new QOpenGLFramebufferObject(
		wdgt2[VRContext::Hand_Left]->size(), GL_TEXTURE_2D);
	wdgt2[VRContext::Hand_Right] = new LeftHandUI();
	wdgt2[VRContext::Hand_Right]->move(0, 0);
	scn2[VRContext::Hand_Right]->addWidget(wdgt2[VRContext::Hand_Left]);
	FBO2[VRContext::Hand_Right] = new QOpenGLFramebufferObject(
		wdgt2[VRContext::Hand_Right]->size(), GL_TEXTURE_2D);

	if (!vr::VROverlay())
	{
		canRun = false;
		return;
	}
	VRContext::forHandsDo([&](uint8_t handIdx) {
		vr::VROverlayError err = vr::VROverlay()->CreateDashboardOverlay(
			OVERLAY_KEY_2[handIdx].data(), OVERLAY_NAME_2[handIdx].data(),
			&overlayHandle2[handIdx], &overlayThumbnailHandle2[handIdx]);
		if (err != vr::VROverlayError_None)
		{
			canRun = false;
			return;
		}
		vr::VROverlay()->SetOverlayWidthInMeters(overlayHandle2[handIdx], 1.5f);
		vr::VROverlay()->SetOverlayInputMethod(overlayHandle2[handIdx],
			vr::VROverlayInputMethod_Mouse);
		});
	if (!canRun) return;

	timer = new QTimer(this);
	QObject::connect(timer, &QTimer::timeout, this, &HandUIHandler::onTimeOut);
	timer->setInterval(20);

	VRContext::forHandsDo([&](uint8_t handIdx) {
		vr::HmdVector2_t vecWindowSize = {
			(float)wdgt2[handIdx]->width(),
			(float)wdgt2[handIdx]->height()
		};
		vr::VROverlay()->SetOverlayMouseScale(overlayHandle2[handIdx], &vecWindowSize);
		});
}

kouek::HandUIHandler::~HandUIHandler()
{

}

void kouek::HandUIHandler::onTimeOut()
{
	if (!vr::VRSystem()) return;
	VRContext::forHandsDo([&](uint8_t handIdx) {
		vr::VREvent_t vrEvent;
		while (vr::VROverlay()->PollNextOverlayEvent(
			overlayHandle2[handIdx], &vrEvent, sizeof(vrEvent)))
		{
			switch (vrEvent.eventType)
			{
			case vr::VREvent_MouseMove:
			{
				QPointF newMouse(vrEvent.data.mouse.x, vrEvent.data.mouse.y);
				QPoint glblNewMouse = newMouse.toPoint();
				QGraphicsSceneMouseEvent mouseEvent(QEvent::GraphicsSceneMouseMove);
				mouseEvent.setWidget(NULL);
				mouseEvent.setPos(newMouse);
				mouseEvent.setScenePos(glblNewMouse);
				mouseEvent.setScreenPos(glblNewMouse);
				mouseEvent.setLastPos(lastMouse);
				mouseEvent.setLastScenePos(
					wdgt2[handIdx]->mapToGlobal(lastMouse.toPoint()));
				mouseEvent.setLastScreenPos(
					wdgt2[handIdx]->mapToGlobal(lastMouse.toPoint()));
				mouseEvent.setButtons(lastMouseBtn);
				mouseEvent.setButton(Qt::NoButton);
				mouseEvent.setModifiers(0);
				mouseEvent.setAccepted(false);

				lastMouse = newMouse;
				QApplication::sendEvent(scn2[handIdx], &mouseEvent);

				if (handIdx == VRContext::Hand_Left)
					onLeftHandSceneChanged(QList<QRectF>());
			}
			break;

			case vr::VREvent_MouseButtonDown:
			{
				Qt::MouseButton button = vrEvent.data.mouse.button == vr::VRMouseButton_Right ? Qt::RightButton : Qt::LeftButton;

				lastMouseBtn |= button;

				QPoint glblLastMouse = lastMouse.toPoint();
				QGraphicsSceneMouseEvent mouseEvent(QEvent::GraphicsSceneMousePress);
				mouseEvent.setWidget(NULL);
				mouseEvent.setPos(lastMouse);
				mouseEvent.setButtonDownPos(button, lastMouse);
				mouseEvent.setButtonDownScenePos(button, glblLastMouse);
				mouseEvent.setButtonDownScreenPos(button, glblLastMouse);
				mouseEvent.setScenePos(glblLastMouse);
				mouseEvent.setScreenPos(glblLastMouse);
				mouseEvent.setLastPos(lastMouse);
				mouseEvent.setLastScenePos(glblLastMouse);
				mouseEvent.setLastScreenPos(glblLastMouse);
				mouseEvent.setButtons(lastMouseBtn);
				mouseEvent.setButton(button);
				mouseEvent.setModifiers(0);
				mouseEvent.setAccepted(false);

				QApplication::sendEvent(scn2[handIdx], &mouseEvent);
			}
			break;

			case vr::VREvent_MouseButtonUp:
			{
				Qt::MouseButton button = vrEvent.data.mouse.button == vr::VRMouseButton_Right ? Qt::RightButton : Qt::LeftButton;
				lastMouseBtn &= ~button;

				QPoint glblLastMouse = lastMouse.toPoint();
				QGraphicsSceneMouseEvent mouseEvent(QEvent::GraphicsSceneMouseRelease);
				mouseEvent.setWidget(NULL);
				mouseEvent.setPos(lastMouse);
				mouseEvent.setScenePos(glblLastMouse);
				mouseEvent.setScreenPos(glblLastMouse);
				mouseEvent.setLastPos(lastMouse);
				mouseEvent.setLastScenePos(glblLastMouse);
				mouseEvent.setLastScreenPos(glblLastMouse);
				mouseEvent.setButtons(lastMouseBtn);
				mouseEvent.setButton(button);
				mouseEvent.setModifiers(0);
				mouseEvent.setAccepted(false);

				QApplication::sendEvent(scn2[handIdx], &mouseEvent);
			}
			break;

			case vr::VREvent_OverlayShown:
				wdgt2[handIdx]->repaint();
			break;

			case vr::VREvent_Quit:
				QApplication::exit();
				break;
			}
		}

		if (overlayThumbnailHandle2[handIdx] != vr::k_ulOverlayHandleInvalid)
		{
			while (vr::VROverlay()->PollNextOverlayEvent(
				overlayThumbnailHandle2[handIdx], &vrEvent, sizeof(vrEvent)))
			{
				switch (vrEvent.eventType)
				{
				case vr::VREvent_OverlayShown:
					wdgt2[handIdx]->repaint();
				break;
				}
			}
		}
		});
}

void kouek::HandUIHandler::onLeftHandSceneChanged(const QList<QRectF>& region)
{
	// skip rendering if the overlay isn't visible
	if ((overlayHandle2[VRContext::Hand_Left] == vr::k_ulOverlayHandleInvalid)
		|| !vr::VROverlay()
		|| (!vr::VROverlay()->IsOverlayVisible(overlayHandle2[VRContext::Hand_Left]) 
			&& !vr::VROverlay()->IsOverlayVisible(overlayThumbnailHandle2[VRContext::Hand_Left])))
		return;

	ctx2[VRContext::Hand_Left]->makeCurrent(offScrnSurf2[VRContext::Hand_Left]);
	FBO2[VRContext::Hand_Left]->bind();
	QOpenGLPaintDevice device(FBO2[VRContext::Hand_Left]->size());
	QPainter painter(&device);
	scn2[VRContext::Hand_Left]->render(&painter);
	FBO2[VRContext::Hand_Left]->release();
}

kouek::QtEventHandler::QtEventHandler(
	EditorWindow* sender,
	std::shared_ptr<AppStates> sharedStates)
	: EventHandler(sharedStates)
{
	QObject::connect(sender, &EditorWindow::closed, [&]() {
		states->canRun = false;
		});
	QObject::connect(sender, &EditorWindow::reloadTFBtnClicked, [&]() {
		kouek::VolumeConfig cfg(std::string(kouek::PROJECT_SOURCE_DIR) + "/cfg/VolumeCfg.json");
		states->renderer->setTransferFunc(cfg.getTF());
		});
	auto increaseRenderTar = [](CompVolumeFAVRRenderer::RenderTarget& renderTar) {
		uint8_t idx = static_cast<uint8_t>(renderTar) + 1;
		if (idx == static_cast<uint8_t>(
			CompVolumeFAVRRenderer::RenderTarget::Last))
			idx = 0;
		renderTar = static_cast<CompVolumeFAVRRenderer::RenderTarget>(idx);
	};
	auto decreaseRenderTar = [](CompVolumeFAVRRenderer::RenderTarget& renderTar) {
		uint8_t idx = static_cast<uint8_t>(renderTar) - 1;
		if (idx > static_cast<uint8_t>(
			CompVolumeFAVRRenderer::RenderTarget::Last))
			idx = static_cast<uint8_t>(
				CompVolumeFAVRRenderer::RenderTarget::Last) - 1;
		renderTar = static_cast<CompVolumeFAVRRenderer::RenderTarget>(idx);
	};
	QObject::connect(sender->getVRView(), &VRView::keyPressed,
		[&](int key, int functionKey) {
			switch (key)
			{
			case Qt::Key_Plus:
				increaseRenderTar(states->renderTar); break;
			case Qt::Key_Minus:
				decreaseRenderTar(states->renderTar); break;
			case Qt::Key_Up:
				moveSteps[2] = +AppStates::moveSensity;
				break;
			case Qt::Key_Down:
				moveSteps[2] = -AppStates::moveSensity;
				break;
			case Qt::Key_Right:
				moveSteps[0] = +AppStates::moveSensity; break;
			case Qt::Key_Left:
				moveSteps[0] = -AppStates::moveSensity; break;
			case Qt::Key_W:
				if (functionKey == Qt::Key_Control)
					subrgnMoveSteps[1] = +AppStates::moveSensity;
				else
					subrgnMoveSteps[2] = +AppStates::moveSensity;
				break;
			case Qt::Key_S:
				if (functionKey == Qt::Key_Control)
					subrgnMoveSteps[1] = -AppStates::moveSensity;
				else
					subrgnMoveSteps[2] = -AppStates::moveSensity;
				break;
			case Qt::Key_D:
				subrgnMoveSteps[0] = +AppStates::moveSensity; break;
			case Qt::Key_A:
				subrgnMoveSteps[0] = -AppStates::moveSensity; break;
			case Qt::Key_1:
				states->game.intrctActMode = InteractionActionMode::SelectVertex; break;
			case Qt::Key_2:
				states->game.intrctActMode = InteractionActionMode::AddPath; break;
			case Qt::Key_3:
				states->game.intrctActMode = InteractionActionMode::AddVertex; break;
			}
		});
}

void kouek::QtEventHandler::update()
{
	if (states->showOverlay2[VRContext::Hand_Left]
		|| states->showOverlay2[VRContext::Hand_Right])
		updateWhenDrawingOverlay();
	else
		updateWhenDrawingCompositor();
}

void kouek::QtEventHandler::updateWhenDrawingOverlay()
{
	if (states->showOverlay2[VRContext::Hand_Left])
	{
		vr::VROverlay()->ShowOverlay(
			handUI.overlayHandle2[VRContext::Hand_Left]);
		if (handUI.timer->isActive())
			handUI.timer->start();
	}
	else if (states->showOverlay2[VRContext::Hand_Right])
	{
		vr::VROverlay()->ShowOverlay(
			handUI.overlayHandle2[VRContext::Hand_Left]);
		if (handUI.timer->isActive())
			handUI.timer->start();
	}
	else
	{
		vr::VROverlay()->HideOverlay(
			handUI.overlayHandle2[VRContext::Hand_Left]);
		vr::VROverlay()->HideOverlay(
			handUI.overlayHandle2[VRContext::Hand_Right]);
		handUI.timer->stop();
	}
}

void kouek::QtEventHandler::updateWhenDrawingCompositor()
{
	if (moveSteps[0] != 0 || moveSteps[1] != 0 || moveSteps[2] != 0)
	{
		states->camera.move(moveSteps[0], moveSteps[1], moveSteps[2]);
		moveSteps = { 0 };
	}
	if (subrgnMoveSteps[0] != 0 || subrgnMoveSteps[1] != 0 || subrgnMoveSteps[2] != 0)
	{
		states->subrgn.center += glm::vec3(subrgnMoveSteps[0],
			subrgnMoveSteps[1], subrgnMoveSteps[2]);
		states->renderer->setSubregion(states->subrgn);
		subrgnMoveSteps = { 0 };
	}
}
