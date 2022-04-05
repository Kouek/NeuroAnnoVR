#ifndef KOUEK_VR_VIEW_H
#define KOUEK_VR_VIEW_H

#include <cassert>
#include <array>

#include <glad/glad.h>
#include <QtGui/qevent.h>
#include <QtWidgets/qopenglwidget.h>

#include <glm/gtc/matrix_transform.hpp>

namespace kouek
{
	class VRView : public QOpenGLWidget
	{
		Q_OBJECT

	private:
		GLuint leftEyeFBO = 0, rightEyeFBO = 0;
		std::array<uint32_t, 2> renderSizePerEye = { 0 };

	public:
		explicit VRView(QWidget* parent = Q_NULLPTR);
		inline void setInputFBOs(
			GLuint leftEyeFBO,
			GLuint rightEyeFBO,
			const std::array<uint32_t, 2>& renderSizePerEye)
		{
			this->leftEyeFBO = leftEyeFBO;
			this->rightEyeFBO = rightEyeFBO;
			this->renderSizePerEye = renderSizePerEye;
		}

	signals:
		void cameraRotated(const glm::mat4& rotation);

	protected:
		void initializeGL() override;
		void paintGL() override;

		void keyPressEvent(QKeyEvent* e) override;
	};
}

#endif // !KOUEK_VR_VIEW_H
