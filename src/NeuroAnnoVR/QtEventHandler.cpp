#include "QtEventHandler.h"

#include <QtWidgets/qapplication.h>
#include <QtWidgets/qgraphicssceneevent.h>

kouek::HandUIHandler::HandUIHandler(
	EditorWindow* glCtxProvider, AppStates* states)
	:glCtxProvider(glCtxProvider), states(states)
{
	glCtxProvider->getVRView()->makeCurrent();

	wdgt2[VRContext::Hand_Left] = new LeftHandUI();
	wdgt2[VRContext::Hand_Right] = new RightHandUI();

	VRContext::forHandsDo([&](uint8_t hndIdx) {
		wdgt2[hndIdx]->move(0, 0);
		scn2[hndIdx] = new QGraphicsScene();
		scn2[hndIdx]->addWidget(wdgt2[hndIdx]);

		FBO2[hndIdx] = new QOpenGLFramebufferObject(
			wdgt2[hndIdx]->size(), GL_TEXTURE_2D);
		glDvc2[hndIdx] = new QOpenGLPaintDevice(
			FBO2[hndIdx]->size());
		pntr2[hndIdx] = new QPainter(glDvc2[hndIdx]);
		});

	QObject::connect(scn2[VRContext::Hand_Left], &QGraphicsScene::changed,
		this, &HandUIHandler::onLeftHandSceneChanged);
	QObject::connect(scn2[VRContext::Hand_Right], &QGraphicsScene::changed,
		this, &HandUIHandler::onRightHandSceneChanged);

	timer = new QTimer(this);
	QObject::connect(timer, &QTimer::timeout, this, &HandUIHandler::onTimeOut);
	timer->setInterval(20);
}

kouek::HandUIHandler::~HandUIHandler()
{
}

void kouek::HandUIHandler::onTimeOut()
{
	static auto isMouseInRange = [&](const glm::vec2& normPos) -> bool {
		if (normPos.x < 0 || normPos.y < 0
			|| normPos.x >= 1.f || normPos.y >= 1.f)
			return false;
		return true;
	};
	while (!states->laserMouseMsgQue.modes.empty())
	{
		auto mode = states->laserMouseMsgQue.modes.front();
		auto normPos = states->laserMouseMsgQue.positions.front();
		states->laserMouseMsgQue.modes.pop();
		states->laserMouseMsgQue.positions.pop();
		switch (mode)
		{
		case LaserMouseMode::MousePressed:
			if (!isMouseInRange(normPos)) break;
			{
				lastMouseButtons |= Qt::LeftButton;

				QPoint glblPt = lastMousePos.toPoint();
				QGraphicsSceneMouseEvent mouseEvent(QEvent::GraphicsSceneMousePress);
				mouseEvent.setWidget(NULL);
				mouseEvent.setPos(lastMousePos);
				mouseEvent.setButtonDownPos(Qt::LeftButton, lastMousePos);
				mouseEvent.setButtonDownScenePos(Qt::LeftButton, glblPt);
				mouseEvent.setButtonDownScreenPos(Qt::LeftButton, glblPt);
				mouseEvent.setScenePos(glblPt);
				mouseEvent.setScreenPos(glblPt);
				mouseEvent.setLastPos(lastMousePos);
				mouseEvent.setLastScenePos(glblPt);
				mouseEvent.setLastScreenPos(glblPt);
				mouseEvent.setButtons(lastMouseButtons);
				mouseEvent.setButton(Qt::LeftButton);
				mouseEvent.setModifiers(0);
				mouseEvent.setAccepted(false);

				QApplication::sendEvent(states->showHandUI2[VRContext::Hand_Left] ?
					scn2[VRContext::Hand_Left] : scn2[VRContext::Hand_Right], &mouseEvent);
			}
			break;
		case LaserMouseMode::MouseMoved:
			if (!isMouseInRange(normPos)) break;
			{
				auto [scn, wdgt] = states->showHandUI2[VRContext::Hand_Left] ?
					std::tuple{ scn2[VRContext::Hand_Left],wdgt2[VRContext::Hand_Left] }
				: std::tuple{ scn2[VRContext::Hand_Right],wdgt2[VRContext::Hand_Right] };

				QPointF newMousePos(normPos.x * scn->width(), normPos.y * scn->height());
				/*ellipse->setPos(newMousePos);*/
				QPoint glblPt = newMousePos.toPoint();
				QGraphicsSceneMouseEvent mouseEvent(QEvent::GraphicsSceneMouseMove);
				mouseEvent.setWidget(NULL);
				mouseEvent.setPos(newMousePos);
				mouseEvent.setScenePos(glblPt);
				mouseEvent.setScreenPos(glblPt);
				mouseEvent.setLastPos(lastMousePos);
				mouseEvent.setLastScenePos(wdgt->mapToGlobal(lastMousePos.toPoint()));
				mouseEvent.setLastScreenPos(wdgt->mapToGlobal(lastMousePos.toPoint()));
				mouseEvent.setButtons(lastMouseButtons);
				mouseEvent.setButton(Qt::NoButton);
				mouseEvent.setModifiers(0);
				mouseEvent.setAccepted(false);

				lastMousePos = newMousePos;
				QApplication::sendEvent(scn, &mouseEvent);
				if (states->showHandUI2[VRContext::Hand_Left])
					onLeftHandSceneChanged(QList<QRectF>());
				else
					onRightHandSceneChanged(QList<QRectF>());
			}
			break;
		case LaserMouseMode::MouseReleased:
			if (!isMouseInRange(normPos)) break;
			{
				lastMouseButtons &= ~Qt::LeftButton;

				QPoint glblPt = lastMousePos.toPoint();
				QGraphicsSceneMouseEvent mouseEvent(QEvent::GraphicsSceneMouseRelease);
				mouseEvent.setWidget(NULL);
				mouseEvent.setPos(lastMousePos);
				mouseEvent.setScenePos(glblPt);
				mouseEvent.setScreenPos(glblPt);
				mouseEvent.setLastPos(lastMousePos);
				mouseEvent.setLastScenePos(glblPt);
				mouseEvent.setLastScreenPos(glblPt);
				mouseEvent.setButtons(lastMouseButtons);
				mouseEvent.setButton(Qt::LeftButton);
				mouseEvent.setModifiers(0);
				mouseEvent.setAccepted(false);

				QApplication::sendEvent(states->showHandUI2[VRContext::Hand_Left] ?
					scn2[VRContext::Hand_Left] : scn2[VRContext::Hand_Right], &mouseEvent);
			}
			break;
		}
	}
}

void kouek::HandUIHandler::onLeftHandSceneChanged(const QList<QRectF>& region)
{
	glCtxProvider->getVRView()->makeCurrent();
	pntr2[VRContext::Hand_Left]->endNativePainting();
	FBO2[VRContext::Hand_Left]->bind();
	scn2[VRContext::Hand_Left]->render(pntr2[VRContext::Hand_Left]);
	FBO2[VRContext::Hand_Left]->release();
	pntr2[VRContext::Hand_Left]->beginNativePainting();
}

void kouek::HandUIHandler::onRightHandSceneChanged(const QList<QRectF>& region)
{
	glCtxProvider->getVRView()->makeCurrent();
	pntr2[VRContext::Hand_Right]->endNativePainting();
	FBO2[VRContext::Hand_Right]->bind();
	scn2[VRContext::Hand_Right]->render(pntr2[VRContext::Hand_Right]);
	FBO2[VRContext::Hand_Right]->release();
	pntr2[VRContext::Hand_Right]->beginNativePainting();
}

kouek::QtEventHandler::QtEventHandler(
	EditorWindow* sender,
	std::shared_ptr<AppStates> sharedStates)
	: EventHandler(sharedStates), handUI(sender, sharedStates.get())
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
	auto computeUITransform = [&](AppStates* states) {
		auto& [R, F, U, PL, PR] = states->camera.getRFUP2();
		auto pos = states->camera.getHeadPos()
			+ AppStates::UITranslateToHead.x * R
			+ AppStates::UITranslateToHead.y * U
			- AppStates::UITranslateToHead.z * F;
		states->handUITransform[0] = glm::vec4{ R, 0 };
		states->handUITransform[1] = glm::vec4{ U, 0 };
		states->handUITransform[2] = glm::vec4{ -F, 0 };
		states->handUITransform[3] = glm::vec4{ pos, 1.f };
	};
	QObject::connect(sender->getVRView(), &VRView::keyPressed,
		[&](int key, int functionKey) {
			switch (key)
			{
				// Render Target Control
			case Qt::Key_Plus:
				increaseRenderTar(states->renderTar); break;
			case Qt::Key_Minus:
				decreaseRenderTar(states->renderTar); break;
				// Camera Control
			case Qt::Key_Up:
				if (functionKey == Qt::Key_Control)
					moveSteps[1] = +AppStates::moveSensity;
				else
					moveSteps[2] = +AppStates::moveSensity;
				break;
			case Qt::Key_Down:
				if (functionKey == Qt::Key_Control)
					moveSteps[1] = -AppStates::moveSensity;
				else
					moveSteps[2] = -AppStates::moveSensity;
				break;
			case Qt::Key_Right:
				moveSteps[0] = +AppStates::moveSensity; break;
			case Qt::Key_Left:
				moveSteps[0] = -AppStates::moveSensity; break;
				// Subregion Control
			case Qt::Key_W:
				if (functionKey == Qt::Key_Control)
					subrgnMoveSteps[1] = +AppStates::subrgnMoveSensity;
				else
					subrgnMoveSteps[2] = +AppStates::subrgnMoveSensity;
				break;
			case Qt::Key_S:
				if (functionKey == Qt::Key_Control)
					subrgnMoveSteps[1] = -AppStates::subrgnMoveSensity;
				else
					subrgnMoveSteps[2] = -AppStates::subrgnMoveSensity;
				break;
			case Qt::Key_D:
				subrgnMoveSteps[0] = +AppStates::subrgnMoveSensity; break;
			case Qt::Key_A:
				subrgnMoveSteps[0] = -AppStates::subrgnMoveSensity; break;
				// Interaction Control
			case Qt::Key_1:
				states->game.intrctActMode = InteractionActionMode::SelectVertex; break;
			case Qt::Key_2:
				states->game.intrctActMode = InteractionActionMode::AddPath; break;
			case Qt::Key_3:
				states->game.intrctActMode = InteractionActionMode::AddVertex; break;
				// Utility
			case Qt::Key_G:
				states->showGizmo = !states->showGizmo; break;
			case Qt::Key_L:
				states->showHandUI2[VRContext::Hand_Left] = !states->showHandUI2[VRContext::Hand_Left];
				states->showHandUI2[VRContext::Hand_Right] = false;
				computeUITransform(states.get());
				break;
			case Qt::Key_R:
				states->showHandUI2[VRContext::Hand_Right] = !states->showHandUI2[VRContext::Hand_Right];
				states->showHandUI2[VRContext::Hand_Left] = false;
				computeUITransform(states.get());
				break;
			case Qt::Key_PageUp:
				states->spacesScale += 1.f;
				states->spacesScaleChanged = true;
				break;
			case Qt::Key_PageDown:
				states->spacesScale -= 1.f;
				states->spacesScaleChanged = true;
				break;
			}
		});
	// LeftHandUI
	QObject::connect(static_cast<LeftHandUI*>(handUI.wdgt2[VRContext::Hand_Left]),
		&LeftHandUI::moveModeBtnsClicked, [&](int id) {
			switch (id)
			{
			case 0:
				states->cameraMountPos = AppStates::CAM_MOUNTED_POS_IN_WANDER;
				states->game.moveMode = MoveMode::Wander;
				break;
			case 1:
				states->cameraMountPos = { states->subrgn.halfW,
					states->subrgn.halfH,states->subrgn.halfD };
				states->game.moveMode = MoveMode::Focus;
				break;
			}
			states->showHandUI2 = { false };
		});
	QObject::connect(static_cast<LeftHandUI*>(handUI.wdgt2[VRContext::Hand_Left]),
		&LeftHandUI::meshAlphaSliderChanged, [&](float alpha) {
			states->meshAlpha = alpha;
		});
	QObject::connect(static_cast<LeftHandUI*>(handUI.wdgt2[VRContext::Hand_Left]),
		&LeftHandUI::spacesScaleChanged, [&](float scale) {
			states->spacesScale = scale;
			states->spacesScaleChanged = true;
		});
	QObject::connect(static_cast<LeftHandUI*>(handUI.wdgt2[VRContext::Hand_Left]),
		&LeftHandUI::tfChanged, [&]() {
			isTFChanged = true;
		});
	// RightHandUI
	QObject::connect(static_cast<RightHandUI*>(handUI.wdgt2[VRContext::Hand_Right]),
		&RightHandUI::interactionActionModeBtnsClicked, [&](int id) {
			switch (id)
			{
			case 0:
				states->game.intrctActMode = InteractionActionMode::SelectVertex; break;
			case 1:
				states->game.intrctActMode = InteractionActionMode::AddPath; break;
			case 2:
				states->game.intrctActMode = InteractionActionMode::AddVertex; break;
			case 3:
				states->game.intrctActMode = InteractionActionMode::DeleteVertex; break;
			case 4:
				states->game.intrctActMode = InteractionActionMode::JoinPath; break;
			}
		});
	QObject::connect(static_cast<RightHandUI*>(handUI.wdgt2[VRContext::Hand_Right]),
		&RightHandUI::interactionModeBtnsClicked, [&](int id) {
			switch (id)
			{
			case 0:
				states->game.intrctParam.mode = 
					CompVolumeFAVRRenderer::InteractionMode::AnnotationBall; break;
			case 1:
				states->game.intrctParam.mode =
					CompVolumeFAVRRenderer::InteractionMode::AnnotationLaser; break;
			}
		});
}

void kouek::QtEventHandler::update()
{
	if (states->showHandUI2[VRContext::Hand_Left]
		|| states->showHandUI2[VRContext::Hand_Right])
	{
		if (!handUI.timer->isActive())
			handUI.timer->start();			
	}
	else
		handUI.timer->stop();

	if (states->showHandUI2[VRContext::Hand_Left])
	{
		if (!lastLeftHandUIShown)
		{
			// every time show LeftHandUI, reload TF
			std::map<uint8_t, std::array<qreal, 4>> tfDat;
			for (auto& tfPt : states->tf.points)
				tfDat.emplace(std::piecewise_construct,
					std::forward_as_tuple(tfPt.key),
					std::forward_as_tuple(tfPt.value));
			static_cast<LeftHandUI*>(handUI.wdgt2[VRContext::Hand_Left])
				->setTFFromTFData(tfDat);

			lastLeftHandUIShown = true;
		}
	}
	else
	{
		if (lastLeftHandUIShown && isTFChanged)
		{
			const auto& tfDat = static_cast<LeftHandUI*>(handUI.wdgt2[VRContext::Hand_Left])
				->getTFData();
			states->tf.points.clear();
			for (const auto& [key, pair] : tfDat)
				states->tf.points.emplace_back(
					vs::TFPoint(key, std::get<std::array<qreal, 4>>(pair)));
			states->renderer->setTransferFunc(states->tf);
			isTFChanged = false;
		}
		lastLeftHandUIShown = false;
	}

	if (moveSteps[0] != 0 || moveSteps[1] != 0 || moveSteps[2] != 0)
	{
		states->camera.move(moveSteps[0], moveSteps[1], moveSteps[2]);
		moveSteps = { 0 };
	}
	if (subrgnMoveSteps[0] != 0 || subrgnMoveSteps[1] != 0 || subrgnMoveSteps[2] != 0)
	{
		states->subrgn.center += glm::vec3(subrgnMoveSteps[0],
			subrgnMoveSteps[1], subrgnMoveSteps[2]);
		states->onSubregionChanged();
		states->renderer->setSubregion(states->subrgn);
		subrgnMoveSteps = { 0 };
	}
}
