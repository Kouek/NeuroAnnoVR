#include "VREventHandler.h"

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

kouek::VREventHandler::VREventHandler(
    std::string_view actionsCfgPath,
    std::shared_ptr<AppStates> sharedStates)
	: EventHandler(sharedStates)
{
	vr::EVRInitError initError;
	HMD = vr::VR_Init(&initError, vr::VRApplication_Scene);
	if (initError != vr::VRInitError_None)
		throw std::exception("VR_Init() FAILED");

	if (!vr::VRCompositor())
		throw std::exception("VRCompositor() FAILED");

	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		states->projection2[eyeIdx] = steamVRMat44ToGLMMat4(
			HMD->GetProjectionMatrix((vr::EVREye)eyeIdx, states->nearClip, states->farClip));
		states->unProjection2[eyeIdx] = Math::inverseProjective(states->projection2[eyeIdx]);

		states->HMDToEye2[eyeIdx] = steamVRMat34ToGLMMat4(HMD->GetEyeToHeadTransform((vr::EVREye)eyeIdx));
		});

    {
        const glm::mat4& headToEye = states->HMDToEye2[vr::Eye_Left];
        glm::vec3 leftEyeToHead = { -headToEye[3][0],-headToEye[3][1],-headToEye[3][2] };
    }
    {
        const glm::mat4& headToEye = states->HMDToEye2[vr::Eye_Right];
        glm::vec3 rightEyeToHead = { -headToEye[3][0],-headToEye[3][1],-headToEye[3][2] };
    }

    if (vr::EVRInputError inputError = vr::VRInput()->SetActionManifestPath(actionsCfgPath.data());
        inputError != vr::VRInputError_None)
        throw std::exception("VRInput()->SetActionManifestPath FAILED");

    auto processInputErr = [](vr::EVRInputError inputErr, int line) {
        if (inputErr == vr::VRInputError_None) return;
        std::string errMsg("VRInput()->GetXXX FAILED, on Line: ");
        errMsg += std::to_string(line);
        throw std::exception(errMsg.c_str());
    };

#define PROCESS_ERR(func) processInputErr(func, __LINE__)

    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/left_axis1_click", &actionLeftTriggerClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/left_axis1_pull", &actionLeftTriggerPull));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/left_south_click", &actionLeftTrackpadSClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/left_north_click", &actionLeftTrackpadNClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/left_west_click", &actionLeftTrackpadWClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/left_east_click", &actionLeftTrackpadEClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/right_axis1_press", &actionRightTriggerPress));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/left_applicationmenu_press", &actionLeftMenu));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/right_south_click", &actionRightTrackpadSClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/right_north_click", &actionRightTrackpadNClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/right_west_click", &actionRightTrackpadWClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/right_east_click", &actionRightTrackpadEClick));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/right_applicationmenu_press", &actionRightMenu));

    PROCESS_ERR(vr::VRInput()->GetActionSetHandle("/actions/focus", &actionsetFocus));

    PROCESS_ERR(vr::VRInput()->GetInputSourceHandle("/user/hand/left", &sourceHands[static_cast<int>(VRContext::HandEnum::Left)]));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/Left_Pose", &actionHandPoses[static_cast<int>(VRContext::HandEnum::Left)]));

    PROCESS_ERR(vr::VRInput()->GetInputSourceHandle("/user/hand/right", &sourceHands[static_cast<int>(VRContext::HandEnum::Right)]));
    PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/Right_Pose", &actionHandPoses[static_cast<int>(VRContext::HandEnum::Right)]));
#undef PROCESS_ERR
}

kouek::VREventHandler::~VREventHandler()
{

}

void kouek::VREventHandler::update()
{
    {
        vr::TrackedDevicePose_t trackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
        vr::VRCompositor()->WaitGetPoses(trackedDevicePoses, vr::k_unMaxTrackedDeviceCount, NULL, 0);
        {
            for (uint32_t devIdx = 0; devIdx < vr::k_unMaxTrackedDeviceCount; ++devIdx)
            {
                if (trackedDevicePoses[devIdx].bPoseIsValid)
                    states->devicePoses[devIdx] = steamVRMat34ToGLMMat4(
                        trackedDevicePoses[devIdx].mDeviceToAbsoluteTracking);
            }
        }
    }
    states->camera.setSelfRotation(states->devicePoses[vr::k_unTrackedDeviceIndex_Hmd]);

    // handle VR input action
    vr::VRActiveActionSet_t activeActionSet = { 0 };
    activeActionSet.ulActionSet = actionsetFocus;
    vr::VRInput()->UpdateActionState(&activeActionSet, sizeof(activeActionSet), 1);
    // digital action
    {
        vr::InputDigitalActionData_t actionData;

        // handle left trackpad
        {
            std::array<float, 3> moveSteps = { 0 };
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadNClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) moveSteps[2] = +moveSensity;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadSClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) moveSteps[2] = -moveSensity;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadWClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) moveSteps[0] = -moveSensity;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadEClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) moveSteps[0] = +moveSensity;
            states->camera.move(moveSteps[0], moveSteps[1], moveSteps[2]);
        }
    }
}
