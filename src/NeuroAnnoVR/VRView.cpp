#include "VRView.h"

kouek::VRView::VRView(QWidget* parent)
	: QOpenGLWidget(parent)
{
	setFocusPolicy(Qt::StrongFocus);
	setCursor(Qt::CrossCursor);
}

void kouek::VRView::initializeGL()
{
	int hasInit = gladLoadGL();
	assert(hasInit);
}

void kouek::VRView::paintGL()
{
	std::array<int, 2> qOGLRenderSize = { width(),height() };
	glBindFramebuffer(GL_READ_FRAMEBUFFER, leftEyeFBO);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, defaultFramebufferObject());
	glBlitFramebuffer(
		0, 0, renderSizePerEye[0], renderSizePerEye[1],
		0, 0, qOGLRenderSize[0] / 2, qOGLRenderSize[1],
		GL_COLOR_BUFFER_BIT, GL_LINEAR);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, rightEyeFBO);
	glBlitFramebuffer(
		0, 0, renderSizePerEye[0], renderSizePerEye[1],
		qOGLRenderSize[0] / 2, 0, qOGLRenderSize[0], qOGLRenderSize[1],
		GL_COLOR_BUFFER_BIT, GL_LINEAR);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

#include <camera/FPSCamera.h>
#include <util/Math.h>

void kouek::VRView::keyPressEvent(QKeyEvent* e)
{
	static auto rotCam = FPSCamera();
	constexpr float rotSensity = 10.f;
	Math::printGLMMat4(rotCam.getViewMat(), "RotateCam");
	switch (e->key())
	{
	case Qt::Key_Up:
		rotCam.rotate(0, +rotSensity); break;
	case Qt::Key_Down:
		rotCam.rotate(0, -rotSensity); break;
	case Qt::Key_Right:
		rotCam.rotate(+rotSensity, 0); break;
	case Qt::Key_Left:
		rotCam.rotate(-rotSensity, 0); break;
	}
	glm::mat4 rot;
	auto [R, F, U, P] = rotCam.getRFUP();
	rot[0][0] = R.x, rot[0][1] = R.y, rot[0][2] = R.z;
	rot[1][0] = U.x, rot[1][1] = U.y, rot[1][2] = U.z;
	rot[2][0] = -F.x, rot[2][1] = -F.y, rot[2][2] = -F.z;
	emit cameraRotated(rot);
}
