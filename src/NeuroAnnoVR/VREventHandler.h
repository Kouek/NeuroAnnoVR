#ifndef KOUEK_VR_EVENT_HANDLER_H
#define KOUEK_VR_EVENT_HANDLER_H

#include <VR/VRContext.h>

#include <camera/DualEyeCamera.h>

namespace kouek
{
    enum class GameMode : int
    {
        FOCUS_MODE = 0,
        WANDER_MODE = 1
    };

    struct StateModel
    {
        bool shouldClose = false;
        DualEyeCamera camera;
    };

	struct VREventHandler
	{
        float moveSensity = .01f;

        vr::VRActionSetHandle_t actionsetFocus = vr::k_ulInvalidActionSetHandle;

        vr::VRActionHandle_t actionLeftTriggerClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionLeftTriggerPull = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionLeftTrackpadSClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionLeftTrackpadNClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionLeftTrackpadWClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionLeftTrackpadEClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionLeftMenu = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionRightTriggerPress = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionRightTrackpadSClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionRightTrackpadNClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionRightTrackpadWClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionRightTrackpadEClick = vr::k_ulInvalidActionHandle;
        vr::VRActionHandle_t actionRightMenu = vr::k_ulInvalidActionHandle;

        std::array<vr::VRActionHandle_t, 2> actionHandPoses = { vr::k_ulInvalidActionHandle };
        std::array<vr::VRInputValueHandle_t, 2> sourceHands = { vr::k_ulInvalidInputValueHandle };

        std::shared_ptr<VRContext> vrCtx;
        std::shared_ptr<StateModel> stateModel;

        VREventHandler(
            std::string_view actionsCfgPath,
            std::shared_ptr<VRContext> vrCtx,
            std::shared_ptr<StateModel> stateModel)
            : vrCtx(vrCtx), stateModel(stateModel)
        {
            {
                glm::mat4& headToEye = vrCtx->HMDToEye2[vr::Eye_Left];
                glm::vec3 leftEyeToHead = { -headToEye[3][0],-headToEye[3][1],-headToEye[3][2] };
                headToEye = vrCtx->HMDToEye2[vr::Eye_Right];
                glm::vec3 rightEyeToHead = { -headToEye[3][0],-headToEye[3][1],-headToEye[3][2] };
            }

            if (vr::EVRInputError inputError = vr::VRInput()->SetActionManifestPath(actionsCfgPath.data());
                inputError != vr::VRInputError_None)
            {
                throw std::exception("()->SetActionManifestPath FAILED");
            }

            auto processInputErr = [](vr::EVRInputError inputErr, int line) {
                if (inputErr == vr::VRInputError_None) return;
                std::string errMsg("VRInput()->GetActionHandle FAILED, on Line: ");
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

            PROCESS_ERR(vr::VRInput()->GetInputSourceHandle("/user/hand/left", &sourceHands[static_cast<int>(VRHandEnum::Left)]));
            PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/Left_Pose", &actionHandPoses[static_cast<int>(VRHandEnum::Left)]));

            PROCESS_ERR(vr::VRInput()->GetInputSourceHandle("/user/hand/right", &sourceHands[static_cast<int>(VRHandEnum::Right)]));
            PROCESS_ERR(vr::VRInput()->GetActionHandle("/actions/focus/in/Right_Pose", &actionHandPoses[static_cast<int>(VRHandEnum::Right)]));
#undef PROCESS_ERR
        }

        void update()
        {
            // handle pose changed action
            stateModel->camera.setSelfRotation(vrCtx->devicePoses[vr::k_unTrackedDeviceIndex_Hmd]);

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
                    stateModel->camera.move(moveSteps[0], moveSteps[1], moveSteps[2]);
                }
            }
        }
	};
}

#endif // !KOUEK_VR_EVENT_HANDLER_H
