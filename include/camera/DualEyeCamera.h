#ifndef KOUEK_DUAL_EYE_CAMERA_H
#define KOUEK_DUAL_EYE_CAMERA_H

#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include <util/VR.h>

namespace kouek
{
	class DualEyeCamera
	{
	private:
		std::array<glm::vec3, 2> dist2;
		std::array<glm::vec3, 2> eyePos2;
		glm::vec3 headPos;
		glm::vec3 right;
		glm::vec3 worldUp, up;
		glm::vec3 forward;
		std::array<glm::mat4x4, 2> view;

	public:
		DualEyeCamera()
			:DualEyeCamera(
				glm::vec3{ 0,0,1.f }, glm::vec3{ 0,0,1.f },
				glm::vec3{ 1.f,1.f,1.f }, glm::vec3{ 0,0,0 }) {}
		DualEyeCamera(
			const glm::vec3& leftEyeToHead, const glm::vec3& rightEyeToHead,
			const glm::vec3& headPos, const glm::vec3& eyeCenter,
			const glm::vec3& up = glm::vec3{ 0,1.f,0 })
			: dist2{ leftEyeToHead, rightEyeToHead }, headPos(headPos)
		{
			this->forward = glm::normalize(eyeCenter - headPos);
			this->worldUp = up;
			updateWithPosForwardUp();
		}
		inline const glm::mat4& getViewMat(uint8_t eyeIdx) const
		{
			return view[eyeIdx];
		}
		inline const glm::vec3& getHeadPos() const
		{
			return headPos;
		}
		inline const glm::vec3& getPos(uint8_t eyeIdx) const
		{
			return eyePos2[eyeIdx];
		}
		inline std::tuple<
			const glm::vec3&, const glm::vec3&, const glm::vec3&,
			const glm::vec3&, const glm::vec3&>
			getRFUP2() const
		{
			return { right, forward, up,
				eyePos2[vr::Eye_Left], eyePos2[vr::Eye_Right] };
		}
		inline void setHeadPos(const glm::vec3& headPos)
		{
			this->headPos = headPos;

			VRContext::forEyesDo([&](uint8_t eyeIdx) {
				eyePos2[eyeIdx] = headPos - dist2[eyeIdx].z * forward
					+ dist2[eyeIdx].y * up
					+ dist2[eyeIdx].x * right;

				view[eyeIdx][3][0] = -glm::dot(right, eyePos2[eyeIdx]);
				view[eyeIdx][3][1] = -glm::dot(up, eyePos2[eyeIdx]);
				view[eyeIdx][3][2] = glm::dot(forward, eyePos2[eyeIdx]);
				});
		}
		inline void setEyeToHead(
			const glm::vec3& leftEyeToHead,
			const glm::vec3& rightEyeToHead)
		{
			dist2 = { leftEyeToHead , rightEyeToHead };

			VRContext::forEyesDo([&](uint8_t eyeIdx) {
				eyePos2[eyeIdx] = headPos - dist2[eyeIdx].z * forward
					+ dist2[eyeIdx].y * up
					+ dist2[eyeIdx].x * right;

				view[eyeIdx][3][0] = -glm::dot(right, eyePos2[eyeIdx]);
				view[eyeIdx][3][1] = -glm::dot(up, eyePos2[eyeIdx]);
				view[eyeIdx][3][2] = glm::dot(forward, eyePos2[eyeIdx]);
				});
		}
		inline void setPosture(const glm::mat4& posture)
		{
			this->right = { posture[0].x,posture[0].y ,posture[0].z };
			this->up = { posture[1].x,posture[1].y ,posture[1].z };
			this->forward = { -posture[2].x,-posture[2].y ,-posture[2].z };
			this->headPos = { posture[3].x,posture[3].y,posture[3].z };

			VRContext::forEyesDo([&](uint8_t eyeIdx) {
				eyePos2[eyeIdx] = headPos - dist2[eyeIdx].z * forward
					+ dist2[eyeIdx].y * up
					+ dist2[eyeIdx].x * right;

				view[eyeIdx][0][0] = right.x;
				view[eyeIdx][1][0] = right.y;
				view[eyeIdx][2][0] = right.z;
				view[eyeIdx][0][1] = up.x;
				view[eyeIdx][1][1] = up.y;
				view[eyeIdx][2][1] = up.z;
				view[eyeIdx][0][2] = -forward.x;
				view[eyeIdx][1][2] = -forward.y;
				view[eyeIdx][2][2] = -forward.z;
				view[eyeIdx][3][0] = -glm::dot(right, eyePos2[eyeIdx]);
				view[eyeIdx][3][1] = -glm::dot(up, eyePos2[eyeIdx]);
				view[eyeIdx][3][2] = glm::dot(forward, eyePos2[eyeIdx]);
				});
		}
		inline void setSelfRotation(const glm::mat3& rotation)
		{
			this->right = { rotation[0].x,rotation[0].y ,rotation[0].z };
			this->up = { rotation[1].x,rotation[1].y ,rotation[1].z };
			this->forward = { -rotation[2].x,-rotation[2].y ,-rotation[2].z };

			VRContext::forEyesDo([&](uint8_t eyeIdx) {
				eyePos2[eyeIdx] = headPos - dist2[eyeIdx].z * forward
					+ dist2[eyeIdx].y * up
					+ dist2[eyeIdx].x * right;

				view[eyeIdx][0][0] = right.x;
				view[eyeIdx][1][0] = right.y;
				view[eyeIdx][2][0] = right.z;
				view[eyeIdx][0][1] = up.x;
				view[eyeIdx][1][1] = up.y;
				view[eyeIdx][2][1] = up.z;
				view[eyeIdx][0][2] = -forward.x;
				view[eyeIdx][1][2] = -forward.y;
				view[eyeIdx][2][2] = -forward.z;
				view[eyeIdx][3][0] = -glm::dot(right, eyePos2[eyeIdx]);
				view[eyeIdx][3][1] = -glm::dot(up, eyePos2[eyeIdx]);
				view[eyeIdx][3][2] = glm::dot(forward, eyePos2[eyeIdx]);
				});
		}
		inline void move(float rightStep, float upStep, float forwardStep)
		{
			headPos += forwardStep * forward + rightStep * right + upStep * up;

			VRContext::forEyesDo([&](uint8_t eyeIdx) {
				eyePos2[eyeIdx] += forwardStep * forward + rightStep * right + upStep * up;
				view[eyeIdx][3][0] = -glm::dot(right, eyePos2[eyeIdx]);
				view[eyeIdx][3][1] = -glm::dot(up, eyePos2[eyeIdx]);
				view[eyeIdx][3][2] = glm::dot(forward, eyePos2[eyeIdx]);
				});
		}

	private:
		inline void updateWithPosForwardUp()
		{
			right = glm::normalize(glm::cross(forward, worldUp));
			up = glm::normalize(glm::cross(right, forward));

			VRContext::forEyesDo([&](uint8_t eyeIdx) {
				eyePos2[eyeIdx] = headPos - dist2[eyeIdx].z * forward
					+ dist2[eyeIdx].y * up
					+ dist2[eyeIdx].x * right;

				view[eyeIdx][0][0] = right.x;
				view[eyeIdx][1][0] = right.y;
				view[eyeIdx][2][0] = right.z;
				view[eyeIdx][0][1] = up.x;
				view[eyeIdx][1][1] = up.y;
				view[eyeIdx][2][1] = up.z;
				view[eyeIdx][0][2] = -forward.x;
				view[eyeIdx][1][2] = -forward.y;
				view[eyeIdx][2][2] = -forward.z;
				view[eyeIdx][3][0] = -glm::dot(right, eyePos2[eyeIdx]);
				view[eyeIdx][3][1] = -glm::dot(up, eyePos2[eyeIdx]);
				view[eyeIdx][3][2] = glm::dot(forward, eyePos2[eyeIdx]);

				view[eyeIdx][3][3] = 1.f;
				view[eyeIdx][0][3] = view[eyeIdx][1][3] = view[eyeIdx][2][3] = 0;
				});
		}
	};
}

#endif // !KOUEK_DUAL_EYE_CAMERA_H
