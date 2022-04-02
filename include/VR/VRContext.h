#ifndef KOUEK_VR_CONTEXT_H
#define KOUEK_VR_CONTEXT_H

#include <exception>
#include <array>
#include <string_view>
#include <functional>

#include <openvr.h>

#include <util/Math.h>

namespace kouek
{
    enum class VRHandEnum : int
    {
        Left = 0,
        Right = 1
    };

	struct VRContext
	{
        std::array<char, vr::k_unMaxTrackedDeviceCount> deviceClasses = { 0 };
		int validPoseCnt = 0;
		float nearClip = 0.01f, farClip = 10.f;
		vr::IVRSystem* HMD;
		std::array<uint32_t, 2> HMDRenderSizePerEye = { 1080,1200 };
		std::array<glm::mat4, vr::k_unMaxTrackedDeviceCount> devicePoses;
		std::array<glm::mat4, 2> projection2, unProjection2;
		std::array<glm::mat4, 2> HMDToEye2;

        static inline glm::mat4 steamVRMat34ToGLMMat4(const vr::HmdMatrix34_t& stMat34)
        {
            return {
                stMat34.m[0][0], stMat34.m[1][0], stMat34.m[2][0], 0.f,
                stMat34.m[0][1], stMat34.m[1][1], stMat34.m[2][1], 0.f,
                stMat34.m[0][2], stMat34.m[1][2], stMat34.m[2][2], 0.f,
                stMat34.m[0][3], stMat34.m[1][3], stMat34.m[2][3], 1.f
            };
        }

        static inline glm::mat4 steamVRMat44ToGLMMat4(const vr::HmdMatrix44_t& stMat44)
        {
            return {
                stMat44.m[0][0], stMat44.m[1][0], stMat44.m[2][0], stMat44.m[3][0],
                stMat44.m[0][1], stMat44.m[1][1], stMat44.m[2][1], stMat44.m[3][1],
                stMat44.m[0][2], stMat44.m[1][2], stMat44.m[2][2], stMat44.m[3][2],
                stMat44.m[0][3], stMat44.m[1][3], stMat44.m[2][3], stMat44.m[3][3]
            };
        }

        static inline std::function<void(std::function<void(uint8_t)>)> forEyesDo =
            [](std::function<void(uint8_t)> f) {
            for (uint8_t eyeIdx = vr::Eye_Left; eyeIdx <= vr::Eye_Right; ++eyeIdx)
                f(eyeIdx);
        };

		VRContext()
		{
            vr::EVRInitError initError;
            HMD = vr::VR_Init(&initError, vr::VRApplication_Scene);
            if (initError != vr::VRInitError_None)
            {
                HMD = NULL;
                throw std::exception("VR_Init FAILED");
            }

            if (!vr::VRCompositor())
            {
                throw std::exception("VRCompositor FAILED");
            }

            forEyesDo([&](uint8_t eyeIdx) {
                projection2[eyeIdx] = steamVRMat44ToGLMMat4(
                    HMD->GetProjectionMatrix((vr::EVREye)eyeIdx, nearClip, farClip));
                unProjection2[eyeIdx] = Math::inverseProjective(projection2[eyeIdx]);

                HMDToEye2[eyeIdx] = steamVRMat34ToGLMMat4(HMD->GetEyeToHeadTransform((vr::EVREye)eyeIdx));
                });
		}

        ~VRContext()
        {
            vr::VR_Shutdown();
        }

        void update()
        {
            // update poses
            vr::TrackedDevicePose_t trackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
            vr::VRCompositor()->WaitGetPoses(trackedDevicePoses, vr::k_unMaxTrackedDeviceCount, NULL, 0);
            {
                for (uint32_t devIdx = 0; devIdx < vr::k_unMaxTrackedDeviceCount; ++devIdx)
                {
                    if (trackedDevicePoses[devIdx].bPoseIsValid)
                        devicePoses[devIdx] = steamVRMat34ToGLMMat4(
                            trackedDevicePoses[devIdx].mDeviceToAbsoluteTracking);
                }
            }
        }
	};
}

#endif // !KOUEK_VR_CONTEXT_H
