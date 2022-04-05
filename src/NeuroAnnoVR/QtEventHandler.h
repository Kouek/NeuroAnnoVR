#ifndef KOUEK_QT_EVENT_HANDLER_H
#define KOUEK_QT_EVENT_HANDLER_H

#include "EventHandler.h"
#include "EditorWindow.h"

namespace kouek
{
	class QtEventHandler : public EventHandler
	{
	public:
		QtEventHandler(
			EditorWindow* sender,
			std::shared_ptr<AppStates> sharedStates)
			: EventHandler(sharedStates)
		{
			QObject::connect(sender, &EditorWindow::closed, [&]() {
				states->canRun = false;
				});
			QObject::connect(sender->getVRView(), &VRView::cameraRotated,
				[&](const glm::mat4& rotation) {
					Math::printGLMMat4(states->camera.getViewMat(0), "States.Camera");
					states->camera.setSelfRotation(rotation);
				});
		}
		void update() override
		{
			// DO NOTHING
		}
	};
}

#endif // !KOUEK_QT_EVENT_HANDLER_H
