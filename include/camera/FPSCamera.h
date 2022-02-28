#ifndef KOUEK_FPS_CAMERA_H
#define KOUEK_FPS_CAMERA_H

#include <glm/gtc/matrix_transform.hpp>

namespace kouek
{
	class FPSCamera
	{
	private:
		glm::vec3 pos;
		glm::vec3 right;
		glm::vec3 up;
		glm::vec3 forward;
		glm::mat4x4 view;

	public:
		FPSCamera()
			:FPSCamera(glm::vec3{ 0,0,1.f }, glm::vec3{ 0,0,0 }) {}
		FPSCamera(
			const glm::vec3& eyePos, const glm::vec3& eyeCenter,
			const glm::vec3& up = glm::vec3{ 0,1.f,0 })
		{
			this->pos = eyePos;
			this->forward = glm::normalize(eyeCenter - eyePos);
			this->up = up;
			updateWithPosForwardUp();
		}
		inline const glm::mat4& getViewMat() const
		{
			return view;
		}
		inline std::tuple<const glm::vec3&, const glm::vec3&, const glm::vec3&>
			getRFU() const
		{
			return { right,forward,up };
		}
		inline void move(const glm::vec3& difPos)
		{
			pos += difPos;
		}
		inline void rotate(float pitchDifDeg, float yawDifDeg)
		{
			float pitchDifRad = glm::radians(pitchDifDeg);
			float yawDifRad = glm::radians(yawDifDeg);
			glm::vec3 tmp = forward;
			forward.y = tmp.y * cosf(pitchDifRad) - tmp.z * sinf(pitchDifRad);
			forward.z = tmp.y * sinf(pitchDifRad) + tmp.z * cosf(pitchDifRad);
			tmp.z = forward.z;
			forward.z = tmp.z * cosf(yawDifRad) - tmp.x * sinf(yawDifRad);
			forward.x = tmp.z * sinf(yawDifRad) + tmp.x * cosf(yawDifRad);
			updateWithPosForwardUp();
		}

	private:
		inline void updateWithPosForwardUp()
		{
			right = glm::normalize(glm::cross(forward, up));
			up = glm::normalize(glm::cross(right, forward));
			view[0][0] = right.x;
			view[1][0] = right.y;
			view[2][0] = right.z;
			view[0][1] = up.x;
			view[1][1] = up.y;
			view[2][1] = up.z;
			view[0][2] = -forward.x;
			view[1][2] = -forward.y;
			view[2][2] = -forward.z;
			view[3][0] = -glm::dot(right, pos);
			view[3][1] = -glm::dot(up, pos);
			view[3][2] = glm::dot(forward, pos);

			view[3][3] = 1.f;
			view[0][3] = view[1][3] = view[2][3] = 0;
		}
	};
}

#endif // !KOUEK_FPS_CAMERA_H
