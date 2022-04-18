#ifndef KOUEK_EVENT_HANDLER_H
#define KOUEK_EVENT_HANDLER_H

#include <array>
#include <openvr.h>

#include <renderer/Renderer.h>

#include <camera/DualEyeCamera.h>
#include <util/Math.h>
#include <util/GLPathRenderer.h>

namespace kouek
{
	constexpr float ANNO_BALL_RADIUS = .01f;
	constexpr float ANNO_BALL_DIAMETER = 2 * ANNO_BALL_RADIUS;
	constexpr float ANNO_BALL_DIST_FROM_HAND = ANNO_BALL_DIAMETER * 2;

	struct Hand
	{
		bool show = false;
		glm::mat4 transform = glm::identity<glm::mat4>();
		std::unique_ptr<VRRenderModel> model;
		std::string modelName;
	};

	enum class GameMode : uint8_t
	{
		Focus = 0,
		Wander
	};

	enum class InteractionActionMode : uint32_t
	{
		SelectVertex = 0x0,
		AddPath = 0x1,
		AddVertex = 0x2,
		DeleteVertex = 0x4,
		SplitSubpath = 0x8,
		JoinPath = 0xC
	};

	struct Game
	{
		GameMode gameMode = GameMode::Wander;
		InteractionActionMode intrctActMode = InteractionActionMode::SelectVertex;
		glm::vec3 intrctPos;
		CompVolumeFAVRRenderer::InteractionParameter intrctParam;

		Game() {
			intrctParam.mode = CompVolumeFAVRRenderer::InteractionMode::AnnotationBall;
			intrctParam.dat.ball.AABBSize = glm::vec3{ ANNO_BALL_DIAMETER };
		}
	};

	struct AppStates
	{
		static inline float moveSensity = .1f;
		static inline float subrgnMoveSensity = .1f;
		static inline float minDistSqrBtwnVerts = .008f;

		bool canRun = true, canVRRun = true;
		float nearClip = 0.01f, farClip = 10.f;
		CompVolumeFAVRRenderer::RenderTarget renderTar = CompVolumeFAVRRenderer::RenderTarget::Image;
		std::array<uint32_t, 2> HMDRenderSizePerEye = { 1080,1080 };
		std::array<glm::mat4, vr::k_unMaxTrackedDeviceCount> devicePoses;
		std::array<glm::mat4, 2> projection2;
		std::array<glm::mat4, 2> unProjection2;
		std::array<glm::mat4, 2> eyeToHMD2;
		DualEyeCamera camera;
		CompVolumeRenderer::Subregion subrgn;
		Hand hand2[2];
		Game game;

		CompVolumeFAVRRenderer* renderer = nullptr;
		GLPathRenderer* pathRenderer = nullptr;

		AppStates()
		{
			projection2[0] = projection2[1] =
				glm::perspectiveFov(
					glm::radians(90.f),
					(float)HMDRenderSizePerEye[0], (float)HMDRenderSizePerEye[1],
					nearClip, farClip);
			unProjection2[0] = unProjection2[1] =
				Math::inverseProjective(projection2[0]);
			eyeToHMD2[vr::Eye_Left] = glm::translate(
				glm::identity<glm::mat4>(), glm::vec3(+.02f, 0, -.01f));
			eyeToHMD2[vr::Eye_Right] = glm::translate(
				glm::identity<glm::mat4>(), glm::vec3(-.02f, 0, -.01f));
			{
				glm::vec3 lftToHead = {
					-eyeToHMD2[vr::Eye_Left][3][0],
					-eyeToHMD2[vr::Eye_Left][3][1],
					-eyeToHMD2[vr::Eye_Left][3][2] };
				glm::vec3 rhtToHead = {
					-eyeToHMD2[vr::Eye_Right][3][0],
					-eyeToHMD2[vr::Eye_Right][3][1],
					-eyeToHMD2[vr::Eye_Right][3][2] };
				camera.setEyeToHead(lftToHead, rhtToHead);
			}
		}
	};

	class EventHandler
	{
	protected:
		std::shared_ptr<AppStates> states;

	public:
		EventHandler(std::shared_ptr<AppStates> sharedStates)
			: states(sharedStates) {}
		virtual void update() = 0;
	};
}

#endif // !KOUEK_EVENT_HANDLER_H
