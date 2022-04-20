#ifndef KOUEK_VR_EVENT_HANDLER_H
#define KOUEK_VR_EVENT_HANDLER_H

#include "EventHandler.h"

namespace kouek
{
	class VREventHandler : public EventHandler
	{
	private:
        std::array<char, vr::k_unMaxTrackedDeviceCount> deviceClasses = { 0 };
        int validPoseCnt = 0;
        vr::IVRSystem* HMD;

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

	public:
		VREventHandler(
			std::string_view actionsCfgPath,
			std::shared_ptr<AppStates> sharedStates);
		~VREventHandler();
		void update() override;

    private:
        void updateWhenDrawingOverlay();
        void updateWhenDrawingCompositor();

        inline void onRightHandTriggerActed(
            const vr::InputDigitalActionData_t& actionDat);
	};
}

#endif // !KOUEK_VR_EVENT_HANDLER_H
