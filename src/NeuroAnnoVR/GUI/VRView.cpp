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

static int functionKey = -1;
void kouek::VRView::keyPressEvent(QKeyEvent* e)
{
	switch (e->key())
	{
	case Qt::Key_Control:
		functionKey = Qt::Key_Control; break;
	case Qt::Key_Shift:
		functionKey = Qt::Key_Shift; break;
	default:
		emit keyPressed(e->key(), functionKey);
	}
}

void kouek::VRView::keyReleaseEvent(QKeyEvent* e)
{
	switch (e->key())
	{
	case Qt::Key_Control:
	case Qt::Key_Shift:
		functionKey = -1;
		break;
	}
}
