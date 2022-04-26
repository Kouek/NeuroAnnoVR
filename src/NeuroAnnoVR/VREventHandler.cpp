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
		states->eyeToHMD2[eyeIdx] = steamVRMat34ToGLMMat4(
            HMD->GetEyeToHeadTransform((vr::EVREye)eyeIdx));
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
        for (uint32_t devIdx = 0; devIdx < vr::k_unMaxTrackedDeviceCount; ++devIdx)
        {
            if (trackedDevicePoses[devIdx].bPoseIsValid)
                states->devicePoses[devIdx] = steamVRMat34ToGLMMat4(
                    trackedDevicePoses[devIdx].mDeviceToAbsoluteTracking);
        }
        auto HMDRealPosture = states->devicePoses[vr::k_unTrackedDeviceIndex_Hmd];
        HMDRealPosture[3] += glm::vec4{ states->cameraMountPos, 0 };
        states->camera.setPosture(HMDRealPosture);
    }
    onHandPosecChanged();

    if (states->showHandUI2[VRContext::Hand_Left]
        || states->showHandUI2[VRContext::Hand_Right])
        updateWhenDrawingUI();
    else
        updateWhenDrawingScene();
}


void kouek::VREventHandler::updateWhenDrawingUI()
{
    // handle VR input action
    vr::VRActiveActionSet_t activeActionSet = { 0 };
    activeActionSet.ulActionSet = actionsetFocus;
    vr::VRInput()->UpdateActionState(&activeActionSet, sizeof(activeActionSet), 1);
    // digital action
    {
        vr::InputDigitalActionData_t actionData;
        // handle left menu
        vr::VRInput()->GetDigitalActionData(actionLeftMenu, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        if (actionData.bActive && actionData.bChanged && actionData.bState)
            states->showHandUI2[VRContext::Hand_Left]
            = states->showHandUI2[VRContext::Hand_Right] = false;
        // handle right menu
        vr::VRInput()->GetDigitalActionData(actionRightMenu, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        if (actionData.bActive && actionData.bChanged && actionData.bState)
            states->showHandUI2[VRContext::Hand_Left]
            = states->showHandUI2[VRContext::Hand_Right] = false;
        // handle right trigger
        vr::VRInput()->GetDigitalActionData(actionRightTriggerPress, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        if (actionData.bActive)
            if (actionData.bState)
                if (actionData.bChanged)
                {
                    states->laserMouseMsgQue.modes.emplace(LaserMouseMode::MousePressed);
                    states->laserMouseMsgQue.positions.emplace(states->laserMouseNormPos);
                }
                else
                {
                    states->laserMouseMsgQue.modes.emplace(LaserMouseMode::MouseMoved);
                    states->laserMouseMsgQue.positions.emplace(states->laserMouseNormPos);
                }
            else if (actionData.bChanged)
            {
                states->laserMouseMsgQue.modes.emplace(LaserMouseMode::MouseReleased);
                states->laserMouseMsgQue.positions.emplace(states->laserMouseNormPos);
            }
            else
            {
                states->laserMouseMsgQue.modes.emplace(LaserMouseMode::MouseMoved);
                states->laserMouseMsgQue.positions.emplace(states->laserMouseNormPos);
            }
    }
}

void kouek::VREventHandler::updateWhenDrawingScene()
{
    // handle VR input action
    vr::VRActiveActionSet_t activeActionSet = { 0 };
    activeActionSet.ulActionSet = actionsetFocus;
    vr::VRInput()->UpdateActionState(&activeActionSet, sizeof(activeActionSet), 1);

    auto computeUITransform = [&]() {
        auto& [R, F, U, PL, PR] = states->camera.getRFUP2();
        auto pos = states->camera.getHeadPos()
            + AppStates::UITranslateToHead.x * R
            + AppStates::UITranslateToHead.y * U
            - AppStates::UITranslateToHead.z * F;
        states->handUITransform[0] = glm::vec4{ R, 0 };
        states->handUITransform[1] = glm::vec4{ U, 0 };
        states->handUITransform[2] = glm::vec4{ -F, 0 };
        states->handUITransform[3] = glm::vec4{ pos, 1.f };
    };

    // digital action
    {
        vr::InputDigitalActionData_t actionData;
        // handle left menu
        vr::VRInput()->GetDigitalActionData(actionLeftMenu, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        if (actionData.bActive && actionData.bChanged && actionData.bState)
        {
            states->showHandUI2[VRContext::Hand_Left] = true;
            states->showHandUI2[VRContext::Hand_Right] = false;
            computeUITransform();
        }
        // handle right menu
        vr::VRInput()->GetDigitalActionData(actionRightMenu, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        if (actionData.bActive && actionData.bChanged && actionData.bState)
        {
            states->showHandUI2[VRContext::Hand_Left] = false;
            states->showHandUI2[VRContext::Hand_Right] = true;
            computeUITransform();
        }
        // handle left trackpad
        {
            static int32_t lastDeg = 0;
            int32_t dltDeg = 0;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadWClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) dltDeg += 10;
            vr::VRInput()->GetDigitalActionData(actionLeftTrackpadEClick, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
            if (actionData.bActive && actionData.bState) dltDeg -= 10;
            if (dltDeg != 0)
            {
                // TODO
            }
        }
        // handle right trigger
        vr::VRInput()->GetDigitalActionData(actionRightTriggerPress, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        onRightHandTriggerPressed(actionData);
    }
    // analog action
    {
        vr::InputAnalogActionData_t actionData;
        // handle left trigger
        vr::VRInput()->GetAnalogActionData(actionLeftTriggerPull, &actionData, sizeof(actionData), vr::k_ulInvalidInputValueHandle);
        onLeftHandTriggerPulled(actionData);
    }
}

void kouek::VREventHandler::onHandPosecChanged()
{
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
            hndPos[0] += states->cameraMountPos.x;
            hndPos[1] += states->cameraMountPos.y;
            hndPos[2] += states->cameraMountPos.z;

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
}

void kouek::VREventHandler::onLeftHandTriggerPulled(
    const vr::InputAnalogActionData_t& actionDat)
{
    static glm::vec4 lastPos = states->hand2[VRContext::Hand_Left].transform[3];
    static bool lastPressed = false;
    if (!actionDat.bActive) return;

    bool pressed = actionDat.x != 0;
    if (pressed && lastPressed)
    {
        glm::vec4 dlt = states->hand2[VRContext::Hand_Left].transform[3];
        dlt = dlt - lastPos;
        dlt = states->camera.getViewMat(vr::Eye_Left) * dlt;
        glm::vec3 drc = dlt;
        drc = glm::normalize(dlt) * AppStates::subrgnMoveSensity;
        states->subrgn.center += drc;
        states->renderer->setSubregion(states->subrgn);
        lastPos = states->hand2[VRContext::Hand_Left].transform[3];
    }
    lastPressed = pressed;
    lastPos = states->hand2[VRContext::Hand_Left].transform[3];
}

void kouek::VREventHandler::onRightHandTriggerPressed(
    const vr::InputDigitalActionData_t& actionDat)
{
    states->game.shouldSelectVertex = false;
    if (!actionDat.bActive) return;

    static bool pressed = false;
    static glm::vec3 lastPos;
    auto isDistBigEnough = [&](const glm::vec3& pos) -> bool {
        glm::vec3 diff = states->game.intrctPos - lastPos;
        float distSqr = glm::dot(diff, diff);
        return distSqr >= AppStates::minDistSqrBtwnVerts;
    };
	switch (states->game.intrctActMode)
	{
    case InteractionActionMode::SelectVertex:
        if (actionDat.bState)
            states->game.shouldSelectVertex = true;
        break;
    case InteractionActionMode::AddPath:
        if (actionDat.bChanged && !actionDat.bState)
        {
            auto pathID = states->pathRenderer->addPath(
                glm::vec3{ 1.f }, states->game.intrctPos);
            states->pathRenderer->endPath();
            states->pathRenderer->startPath(pathID);
            lastPos = states->game.intrctPos;
        }
        break;
	case InteractionActionMode::AddVertex:
        if (actionDat.bChanged && actionDat.bState)
        {
            auto id = states->pathRenderer->addSubPath();
            states->pathRenderer->startSubPath(id);
            pressed = true;
        }
        else if (actionDat.bChanged && !actionDat.bState)
        {
            GLuint id = states->pathRenderer->getSelectedVertID();
            id = states->pathRenderer->addVertex(states->game.intrctPos);
            states->pathRenderer->endSubPath();
            states->pathRenderer->startVertex(id);
            lastPos = states->game.intrctPos;
            pressed = false;
        }
        else if (actionDat.bState && isDistBigEnough(states->game.intrctPos))
        {
            auto id = states->pathRenderer->addVertex(states->game.intrctPos);
            states->pathRenderer->startVertex(id);
            lastPos = states->game.intrctPos;
        }
		break;
	default:
		break;
	}
}
