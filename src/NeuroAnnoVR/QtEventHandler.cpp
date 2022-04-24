#include "QtEventHandler.h"

#include <QtWidgets/qapplication.h>
#include <QtWidgets/qgraphicssceneevent.h>

kouek::HandUIHandler::HandUIHandler(EditorWindow* glCtxProvider)
	:glCtxProvider(glCtxProvider)
{
	glCtxProvider->getVRView()->makeCurrent();

	wdgt2[VRContext::Hand_Left] = new LeftHandUI();
	wdgt2[VRContext::Hand_Right] = new LeftHandUI();

	VRContext::forHandsDo([&](uint8_t hndIdx) {
		wdgt2[hndIdx]->move(0, 0);
		scn2[hndIdx] = new QGraphicsScene();
		scn2[hndIdx]->addWidget(wdgt2[VRContext::Hand_Left]);

		FBO2[hndIdx] = new QOpenGLFramebufferObject(
			wdgt2[hndIdx]->size(), GL_TEXTURE_2D);
		glDvc2[hndIdx] = new QOpenGLPaintDevice(
			FBO2[VRContext::Hand_Left]->size());
		pntr2[hndIdx] = new QPainter(glDvc2[hndIdx]);
		});

	QObject::connect(scn2[0], &QGraphicsScene::changed,
		this, &HandUIHandler::onLeftHandSceneChanged);

	timer = new QTimer(this);
	QObject::connect(timer, &QTimer::timeout, this, &HandUIHandler::onTimeOut);
	timer->setInterval(20);
}

kouek::HandUIHandler::~HandUIHandler()
{

}

void kouek::HandUIHandler::onTimeOut()
{
	onLeftHandSceneChanged(QList<QRectF>());
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

kouek::QtEventHandler::QtEventHandler(
	EditorWindow* sender,
	std::shared_ptr<AppStates> sharedStates)
	: EventHandler(sharedStates), handUI(sender)
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
				break;
			case Qt::Key_R:
				states->showHandUI2[VRContext::Hand_Right] = !states->showHandUI2[VRContext::Hand_Right];
				states->showHandUI2[VRContext::Hand_Left] = false;
				break;
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
