#ifndef KOUEK_QT_EVENT_HANDLER_H
#define KOUEK_QT_EVENT_HANDLER_H

#include "EventHandler.h"
#include "EditorWindow.h"

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
			QObject::connect(sender->getVRView(), &VRView::keyPressed,
				[&](int key) {
					switch (key)
					{
					case Qt::Key_Up:
						moveSteps[2] = +AppStates::moveSensity; break;
					case Qt::Key_Down:
						moveSteps[2] = -AppStates::moveSensity; break;
					case Qt::Key_Right:
						moveSteps[0] = +AppStates::moveSensity; break;
					case Qt::Key_Left:
						moveSteps[0] = -AppStates::moveSensity; break;
					case Qt::Key_W:
						subrgnMoveSteps[2] = +AppStates::moveSensity; break;
					case Qt::Key_S:
						subrgnMoveSteps[2] = -AppStates::moveSensity; break;
					case Qt::Key_D:
						subrgnMoveSteps[0] = +AppStates::moveSensity; break;
					case Qt::Key_A:
						subrgnMoveSteps[0] = -AppStates::moveSensity; break;
					}
				});
		}
		void update() override
		{
			{
				states->camera.move(moveSteps[0], moveSteps[1], moveSteps[2]);
				moveSteps = { 0 };
			}
			if (subrgnMoveSteps[0] != 0 || subrgnMoveSteps[1] != 0 || subrgnMoveSteps[2] != 0)
			{
				states->subrgn.center += glm::vec3(subrgnMoveSteps[0],
					subrgnMoveSteps[1], subrgnMoveSteps[2]);
				subrgnMoveSteps = { 0 };
				states->subrgnChanged = true;
			}
		}
	};
}

#endif // !KOUEK_QT_EVENT_HANDLER_H
