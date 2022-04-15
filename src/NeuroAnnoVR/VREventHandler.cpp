#include "VREventHandler.h"

#include <spdlog/spdlog.h>

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

static std::string getTrackedDeviceString(vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop,
    vr::TrackedPropertyError* peError = NULL)
{
    uint32_t unRequiredBufferLen = vr::VRSystem()->GetStringTrackedDeviceProperty(unDevice, prop, NULL, 0, peError);
    if (unRequiredBufferLen == 0) return "";

    char* pchBuffer = new char[unRequiredBufferLen];
    unRequiredBufferLen =
        vr::VRSystem()->GetStringTrackedDeviceProperty(unDevice, prop, pchBuffer, unRequiredBufferLen, peError);
    std::string sResult = pchBuffer;
    delete[] pchBuffer;
    return sResult;
}

static std::tuple<vr::RenderModel_t*, vr::RenderModel_TextureMap_t*> getRenderModelAndTex(
    const char* modelName)
{
    vr::RenderModel_t* model = nullptr;
    vr::EVRRenderModelError error;
    while (true)
    {
        error = vr::VRRenderModels()->LoadRenderModel_Async(modelName, &model);
        if (error != vr::VRRenderModelError_Loading) break;
        // block until loading model finished
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (error != vr::VRRenderModelError_None)
    {
        spdlog::error("{0} - Load render model FAILED: Model Name:{1}, {2}", __FUNCTION__, modelName,
            vr::VRRenderModels()->GetRenderModelErrorNameFromEnum(error));
        return { nullptr,nullptr };
    }

    vr::RenderModel_TextureMap_t* tex;
    while (1)
    {
        error = vr::VRRenderModels()->LoadTexture_Async(model->diffuseTextureId, &tex);
        if (error != vr::VRRenderModelError_Loading) break;
        // block until loading texture finished
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (error != vr::VRRenderModelError_None)
    {
        spdlog::error("{0} - Load render model's texture FAILED: Model Name:{1}, Texture ID: {2}, {3}", __FUNCTION__,
            modelName, model->diffuseTextureId,
            vr::VRRenderModels()->GetRenderModelErrorNameFromEnum(error));
        return { nullptr,nullptr };
    }
    return { model,tex };
}

kouek::VREventHandler::VREventHandler(
    std::string_view actionsCfgPath,
    std::shared_ptr<AppStates> sharedStates)
	: EventHandler(sharedStates)
{
	vr::EVRInitError initError;
	HMD = vr::VR_Init(&initError, vr::VRApplication_Scene);
	if (initError != vr::VRInitError_None)
		throw std::runtime_error("VR_Init() FAILED");

	if (!vr::VRCompositor())
		throw std::runtime_error("VRCompositor() FAILED");

    // HMD->GetRecommendedRenderTargetSize() is too costly
    states->HMDRenderSizePerEye[0] = 1080;
    states->HMDRenderSizePerEye[1] = 1080;

	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		states->projection2[eyeIdx] = steamVRMat44ToGLMMat4(
			HMD->GetProjectionMatrix((vr::EVREye)eyeIdx, states->nearClip, states->farClip));
		states->unProjection2[eyeIdx] = Math::inverseProjective(states->projection2[eyeIdx]);

		states->eyeToHMD2[eyeIdx] = steamVRMat34ToGLMMat4(HMD->GetEyeToHeadTransform((vr::EVREye)eyeIdx));
		});

    {
        glm::vec3 leftEyeToHead, rightEyeToHead;
        {
            const glm::mat4& eyeToHead = states->eyeToHMD2[vr::Eye_Left];
            leftEyeToHead = { eyeToHead[3][0],eyeToHead[3][1],eyeToHead[3][2] };
        }
        {
            const glm::mat4& eyeToHead = states->eyeToHMD2[vr::Eye_Right];
            rightEyeToHead = { eyeToHead[3][0],eyeToHead[3][1],eyeToHead[3][2] };
        }
        states->camera.setEyeToHead(leftEyeToHead, rightEyeToHead);
    }

    if (vr::EVRInputError inputError = vr::VRInput()->SetActionManifestPath(actionsCfgPath.data());
        inputError != vr::VRInputError_None)
        throw std::runtime_error("VRInput()->SetActionManifestPath FAILED");

    auto processInputErr = [](vr::EVRInputError inputErr, int line) {
        if (inputErr == vr::VRInputError_None) return;
        std::string errMsg("VRInput()->GetXXX FAILED, on Line: ");
        errMsg += std::to_string(line);
        throw std::runtime_error(errMsg.c_str());
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
    // handle HMD pose changed
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
        states->camera.setSelfRotation(states->devicePoses[vr::k_unTrackedDeviceIndex_Hmd]);
    }
    auto& HMDPose = states->devicePoses[vr::k_unTrackedDeviceIndex_Hmd];

    // hanlde hand pose changed
    VRContext::forHandsDo([&](uint8_t hndIdx) {
        vr::InputPoseActionData_t poseData;
        if (vr::VRInput()->GetPoseActionDataForNextFrame(
            actionHandPoses[hndIdx], vr::TrackingUniverseStanding,
            &poseData, sizeof(poseData), vr::k_ulInvalidInputValueHandle)
            != vr::VRInputError_None
            || !poseData.bActive || !poseData.pose.bPoseIsValid)
            states->hand2[hndIdx].show = false;
        else
        {
            states->hand2[hndIdx].show = true;
            states->hand2[hndIdx].transform = steamVRMat34ToGLMMat4(
                poseData.pose.mDeviceToAbsoluteTracking);
            auto& hndPos = states->hand2[hndIdx].transform[3];
            hndPos[0] = hndPos[0] - HMDPose[3][0] + states->camera.getHeadPos().x;
            hndPos[1] = hndPos[1] - HMDPose[3][1] + states->camera.getHeadPos().y;
            hndPos[2] = hndPos[2] - HMDPose[3][2] + states->camera.getHeadPos().z;

            vr::InputOriginInfo_t originInfo;
            if (vr::VRInput()->GetOriginTrackedDeviceInfo(
                poseData.activeOrigin, &originInfo, sizeof(originInfo)) ==
                vr::VRInputError_None &&
                originInfo.trackedDeviceIndex != vr::k_unTrackedDeviceIndexInvalid)
            {
                std::string modelName =
                    getTrackedDeviceString(originInfo.trackedDeviceIndex, vr::Prop_RenderModelName_String);
                // when name changed, change render model
                if (modelName != states->hand2[hndIdx].modelName)
                {
                    states->hand2[hndIdx].modelName = modelName;
                    auto [model, tex] = getRenderModelAndTex(modelName.c_str());
                    if (model != nullptr && tex != nullptr)
                        states->hand2[hndIdx].model = std::make_unique<VRRenderModel>(
                            *model, *tex);
                }
            }
        }
        });

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
            if (actionData.bActive && actionData.bState) moveSteps[2] = +AppStates::moveSensity;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadSClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) moveSteps[2] = -AppStates::moveSensity;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadWClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) moveSteps[0] = -AppStates::moveSensity;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadEClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) moveSteps[0] = +AppStates::moveSensity;
            states->camera.move(moveSteps[0], moveSteps[1], moveSteps[2]);
        }

        // handle right trigger
        vr::VRInput()->GetDigitalActionData(actionRightTriggerPress, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        if (actionData.bActive && actionData.bState && actionData.bChanged)
            handleRightHandTrigger();
    }
}

void kouek::VREventHandler::handleRightHandTrigger()
{
    try
    {
        switch (states->game.intrctActMode)
        {
        case InteractionActionMode::AddVertex:
            states->pathManager->addVertex(states->game.intrctPos);
            break;
        default:
            break;
        }
    }
    catch (std::exception& e)
    {
        spdlog::info("{0}", e.what());
    }
}
