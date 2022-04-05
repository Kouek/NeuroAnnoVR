#ifndef KOUEK_VR_H
#define KOUEK_VR_H

#include <functional>

#include <openvr.h>

namespace
{
	class VRContext
	{
	public:
		enum class HandEnum : int
		{
			Left = 0,
			Right = 1
		};

		static inline std::function<void(std::function<void(uint8_t)>)> forEyesDo =
			[](std::function<void(uint8_t)> func) {
			for (uint8_t eyeIdx = vr::Eye_Left; eyeIdx <= vr::Eye_Right; ++eyeIdx)
				func(eyeIdx);
		};
	};
}

#endif // !KOUEK_VR_H
