#ifndef KOUEK_QT_EVENT_HANDLER_H
#define KOUEK_QT_EVENT_HANDLER_H

#include "EventHandler.h"
#include "EditorWindow.h"

#include <CMakeIn.h>
#include <util/VolumeCfg.h>

namespace kouek
{
	class QtEventHandler : public EventHandler
	{
	private:
		std::array<float, 3> moveSteps = { 0 };
		std::array<float, 3> subrgnMoveSteps = { 0 };

	public:
		QtEventHandler(
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
		void update() override
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
	};
}

#endif // !KOUEK_QT_EVENT_HANDLER_H
