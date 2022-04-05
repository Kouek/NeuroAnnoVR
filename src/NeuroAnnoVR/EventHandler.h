#ifndef KOUEK_EVENT_HANDLER_H
#define KOUEK_EVENT_HANDLER_H

#include <array>
#include <openvr.h>

#include <camera/DualEyeCamera.h>
#include <util/Math.h>

namespace kouek
{
	struct AppStates
	{
		bool canRun = true, canVRRun = true;
		DualEyeCamera camera;
		std::array<uint32_t, 2> HMDRenderSizePerEye = { 800,600 };
		std::array<glm::mat4, vr::k_unMaxTrackedDeviceCount> devicePoses;
		float nearClip = 0.01f, farClip = 10.f;
		std::array<glm::mat4, 2> projection2;
		std::array<glm::mat4, 2> unProjection2;
		std::array<glm::mat4, 2> HMDToEye2;

		AppStates()
		{
			projection2[0] = projection2[1] =
				glm::perspectiveFov(
					glm::radians(90.f),
					(float)HMDRenderSizePerEye[0], (float)HMDRenderSizePerEye[1],
					nearClip, farClip);
			unProjection2[0] = unProjection2[1] =
				Math::inverseProjective(projection2[0]);
			HMDToEye2[vr::Eye_Left] = glm::translate(
				glm::identity<glm::mat4>(), glm::vec3(+.02f, 0, -.01f));
			HMDToEye2[vr::Eye_Right] = glm::translate(
				glm::identity<glm::mat4>(), glm::vec3(-.02f, 0, -.01f));
			{
				glm::vec3 lftToHead = {
					-HMDToEye2[vr::Eye_Left][3][0],
					-HMDToEye2[vr::Eye_Left][3][1],
					-HMDToEye2[vr::Eye_Left][3][2] };
				glm::vec3 rhtToHead = {
					-HMDToEye2[vr::Eye_Right][3][0],
					-HMDToEye2[vr::Eye_Right][3][1],
					-HMDToEye2[vr::Eye_Right][3][2] };
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
