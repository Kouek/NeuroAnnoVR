#ifndef KOUEK_EVENT_HANDLER_H
#define KOUEK_EVENT_HANDLER_H

#include <array>
#include <queue>

#include <openvr.h>

#include <renderer/Renderer.h>
#include <renderer/GLPathRenderer.h>

#include <camera/DualEyeCamera.h>
#include <util/Math.h>

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

	enum class MoveMode : uint8_t
	{
		Focus = 0,
		Wander
	};

	enum class InteractionActionMode : uint32_t
	{
		SelectVertex = 0x1,
		AddPath = 0x2,
		AddVertex = 0x4,
		DeleteVertex = 0x8,
		JoinPath = 0x10
	};

	enum class LaserMouseMode : uint8_t
	{
		None = 0,
		MouseMoved,
		MousePressed,
		MouseReleased
	};

	struct Game
	{
		bool shouldSelectVertex = false;
		MoveMode moveMode = MoveMode::Wander;
		InteractionActionMode intrctActMode = InteractionActionMode::SelectVertex;
		glm::vec3 intrctPos;
		std::array<GLuint, 2> intrctLineVertID2;
		std::array<std::array<glm::vec3, 2>, 2> intrctLineDat2;
		CompVolumeFAVRRenderer::InteractionParameter intrctParam;

		Game() {
			intrctParam.mode = CompVolumeFAVRRenderer::InteractionMode::AnnotationBall;
			intrctParam.dat.ball.AABBSize = glm::vec3{ ANNO_BALL_DIAMETER };
		}
	};

	struct LaserMouseMessageQue
	{
		std::queue<glm::vec2> positions;
		std::queue<LaserMouseMode> modes;
	};

	struct AppStates
	{
		static inline constexpr glm::vec3 CAM_MOUNTED_POS_IN_WANDER = { 0, -.6f, 0 };

		static inline float moveSensity = .1f;
		static inline float subrgnMoveSensity = .002f;
		static inline float subrgnMoveSensityFine = .001f;
		static inline float minDistSqrBtwnVerts = .004f;
		static inline glm::vec3 UITranslateToHead = glm::vec3{ 0,0,-2.f };

		bool canRun = true, canVRRun = true;
		std::array<bool, 2> showHandUI2 = { false };
		bool showGizmo = false;
		bool spacesScaleChanged = false;
		float nearClip = 0.01f, farClip = 10.f;
		CompVolumeFAVRRenderer::RenderTarget renderTar = CompVolumeFAVRRenderer::RenderTarget::Image;

		std::array<uint32_t, 2> HMDRenderSizePerEye = { 1080,1080 };
		std::array<glm::mat4, vr::k_unMaxTrackedDeviceCount> devicePoses;
		std::array<glm::mat4, 2> projection2;
		std::array<glm::mat4, 2> unProjection2;
		std::array<glm::mat4, 2> eyeToHMD2;
		
		glm::mat4 scaleWdToVx, scaleVxToWd;
		glm::mat4 fromWdToVxSp, fromVxToWdSp;

		float meshAlpha = .5f;
		float spacesScale = 5.f, lastSpacesScale = 5.f;
		glm::vec2 laserMouseNormPos;
		glm::vec3 cameraMountPos;
		glm::mat4 handUITransform;
		glm::mat4 gizmoTransform;

		CompVolumeFAVRRenderer* renderer = nullptr;
		GLPathRenderer* pathRenderer = nullptr;

		DualEyeCamera camera;
		vs::TransferFunc tf;
		CompVolumeRenderer::Subregion subrgn;
		Hand hand2[2];
		Game game;
		LaserMouseMessageQue laserMouseMsgQue;

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
		void onSubregionChanged()
		{
			glm::vec3 oriCntr = {
				subrgn.halfW / scaleVxToWd[0][0],
				subrgn.halfH / scaleVxToWd[1][1],
				subrgn.halfD / scaleVxToWd[2][2] };
			glm::vec3 cntr = {
				subrgn.center.x / scaleVxToWd[0][0],
				subrgn.center.y / scaleVxToWd[1][1],
				subrgn.center.z / scaleVxToWd[2][2] };
			fromVxToWdSp = glm::translate(glm::identity<glm::mat4>(),
				oriCntr - cntr);
			fromVxToWdSp = scaleVxToWd * fromVxToWdSp;

			oriCntr = { subrgn.halfW,
				subrgn.halfH, subrgn.halfD };
			cntr = { subrgn.center.x,
				subrgn.center.y, subrgn.center.z };
			fromWdToVxSp = glm::translate(glm::identity<glm::mat4>(),
				cntr - oriCntr);
			fromWdToVxSp = scaleWdToVx * fromWdToVxSp;
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
