#ifndef KOUEK_MAIN_APP_H
#define KOUEK_MAIN_APP_H

#include <CMakeIn.h>
#include <renderer/Renderer.h>
#include <renderer/GLObjRenderer.h>

#include <util/SWC.h>

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
		static inline constexpr auto bkGrndCol = glm::vec3{ .5f,.5f,.5f };
		static inline constexpr auto identity = glm::identity<glm::mat4>();

		std::shared_ptr<AppStates> states;
		std::unique_ptr<QApplication> qtApp;
		std::unique_ptr<EditorWindow> editorWindow;
		std::unique_ptr<VREventHandler> vrEvntHndler;
		std::unique_ptr<QtEventHandler> qtEvntHndler;
		std::unique_ptr<FileSWC> swc;
		std::unique_ptr<GLPathRenderer> pathRenderer;
		std::unique_ptr<Shaders> shaders;

		std::array<glm::mat4, 2> VP2;
		std::array<std::array<glm::mat4, 2>, 2> handUIMVP2;
		std::array<glm::mat4, 2> gizmoMVP2;
		std::array<std::array<glm::mat4, 2>, 2> handMVP22;
		std::array<glm::mat4, 2> pathMVP2;

		struct
		{
			glm::vec3 intersectPos;
			glm::vec4 projectedPos;
			std::unique_ptr<WireFrame> model;
		}laser;

		struct
		{
			glm::vec4 projectedPos;
			glm::vec2 screenPos;
			std::array<glm::mat4, 2> MVP2;
			glm::mat4 transform;
			std::unique_ptr<WireFrame> model;
		}ball;

		VolumeRenderType volumeRender;

		struct
		{
			std::unique_ptr<WireFrame> model;
		}gizmo;

		struct
		{
			std::unique_ptr<Point> model;
		}intrctPoint;

		struct
		{
			GLuint VAO, VBO, EBO;
		}screenQuad, handUIQuad[2];

		struct
		{
			GLuint FBO, colorTex, depthRBO;
		}depthFramebuffer2[2]{ 0 }, colorFramebuffer2[2]{ 0 },
			submitFramebuffer2[2]{ 0 }, pathSelectFramebuffer{ 0 };

	public:
		MainApp(int argc, char** argv);
		~MainApp();
		int run();

	private:
		void drawUI();
		void drawScene();
	};
}

#endif // !KOUEK_MAIN_APP_H
