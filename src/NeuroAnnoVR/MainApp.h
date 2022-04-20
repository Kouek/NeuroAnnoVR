#ifndef KOUEK_MAIN_APP_H
#define KOUEK_MAIN_APP_H

#include <CMakeIn.h>
#include <renderer/Renderer.h>
#include <util/RenderObj.h>

#include <QtWidgets/qapplication.h>

#include "VREventHandler.h"
#include "QtEventHandler.h"
#include "Shaders.h"

namespace kouek
{
	struct VolumeRenderType
	{
		uint32_t noPadBlkLen;
		std::array<GLuint, 2> tex2;
		std::shared_ptr<vs::CompVolume> volume;
		std::unique_ptr<kouek::CompVolumeFAVRRenderer> renderer;
	};

	class MainApp
	{
	private:
		std::shared_ptr<AppStates> states;
		std::unique_ptr<QApplication> qtApp;
		std::unique_ptr<EditorWindow> editorWindow;
		std::unique_ptr<EventHandler> vrEvntHndler;
		std::unique_ptr<EventHandler> qtEvntHndler;
		std::unique_ptr<GLPathRenderer> pathRenderer;
		std::unique_ptr<Shaders> shaders;

		std::array<glm::mat4, 2> VP2;
		std::array<glm::mat4, 2> gizmoMVP2;
		std::array<std::array<glm::mat4, 2>, 2> handMVP22;

		union
		{
			struct
			{
				glm::vec4 projectedPos;
				glm::vec2 screenPos;
				std::array<glm::mat4, 2> MVP2;
				glm::mat4 transform;
			}ball;
		}anno;

		struct
		{
			std::unique_ptr<WireFrame> model;
		}annoBall;

		VolumeRenderType volumeRender;

		struct
		{
			glm::mat4 transform = glm::identity<glm::mat4>();
			std::unique_ptr<WireFrame> model;
		}gizmo;

		struct
		{
			std::unique_ptr<Point> model;
		}intrctPoint;

		struct
		{
			GLuint VAO, VBO, EBO;
		}screenQuad;

		struct
		{
			GLuint FBO, colorTex, depthRBO;
		}depthFramebuffer2[2]{ 0 }, colorFramebuffer2[2]{ 0 },
			submitFramebuffer2[2]{ 0 }, pathSelectFramebuffer{ 0 };

	public:
		MainApp(int argc, char** argv);
		~MainApp();
		int run();
	};
}

#endif // !KOUEK_MAIN_APP_H
