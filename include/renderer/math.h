#ifndef KOUEK_MATH_H
#define KOUEK_MATH_H

#include <glm/gtc/matrix_transform.hpp>

namespace kouek
{
	class Math
	{
	public:
		static const inline glm::mat4 IDENTITY = glm::identity<glm::mat4>();
		static glm::mat4 inverseProjective(const glm::mat4 projection)
		{

		}
	};
}

#endif // !KOUEK_MATH_H
