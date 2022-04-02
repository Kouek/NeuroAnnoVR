#include <CMakeIn.h>

#include <renderer/Renderer.h>

#include <QtWidgets/qapplication.h>
#include <QtWidgets/qopenglwidget.h>
#include <QtGui/qopenglshaderprogram.h>

#include <spdlog/spdlog.h>

#include "VREventHandler.h"

#include <util/VolumeCfg.h>
#include <util/RenderObj.h>

#define GL_CHECK \
         {       \
            GLenum gl_err; \
            if((gl_err=glGetError())!=GL_NO_ERROR){     \
            spdlog::error("OpenGL error: {0} caused before  on line {1} of file:{2}",static_cast<unsigned int>(gl_err),__LINE__,__FILE__);     \
            }\
         }

using namespace kouek;

static GLuint createPlainTexture(uint32_t w, uint32_t h)
{
	GLuint tex;
	glGenTextures(1, &tex);

	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, (const void*)0);
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}

static std::tuple<GLuint, GLuint, GLuint> createFrambuffer(uint32_t w, uint32_t h, bool useMultiSample = false)
{
	GLuint FBO, colorTex, depthRBO;

	glGenFramebuffers(1, &FBO);
	glGenTextures(1, &colorTex);
	glGenRenderbuffers(1, &depthRBO);

	if (useMultiSample)
	{
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, colorTex);
		glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, w, h, true);
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);

		glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT, w, h);

		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D_MULTISAMPLE, colorTex, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
	}
	else
	{
		colorTex = createPlainTexture(w, h);

		glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, colorTex, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_RENDERBUFFER, depthRBO);
	}

	int FBOStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	assert(FBOStatus == GL_FRAMEBUFFER_COMPLETE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return { FBO,colorTex,depthRBO };
}

static std::tuple<GLuint, GLuint, GLuint> createScreenQuad()
{
	GLuint VAO, VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	{
		std::vector<std::array<GLfloat, 8>> verts;
		verts.push_back({ -1.f,-1.f,+0.f, +0.f,+0.f,+0.f, +0.f,+0.f });
		verts.push_back({ +1.f,-1.f,+0.f, +0.f,+0.f,+0.f, +1.f,+0.f });
		verts.push_back({ -1.f,+1.f,+0.f, +0.f,+0.f,+0.f, +0.f,+1.f });
		verts.push_back({ +1.f,+1.f,+0.f, +0.f,+0.f,+0.f, +1.f,+1.f });

		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8 * verts.size(),
			verts.data()->data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 8, (const void*)(0));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 8, (const void*)(sizeof(GLfloat) * 3));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 8, (const void*)(sizeof(GLfloat) * 6));
	}

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	{
		GLushort idxes[] = { 0, 1, 3, 0, 3, 2 };
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * 6, idxes, GL_STATIC_DRAW);
	}
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return { VAO, VBO, EBO };
}

static std::unique_ptr<WireFrame> createGizmo()
{
	std::vector<GLfloat> verts = {
		// axis
		+0.0f,+0.0f,+0.0f, +1.0f,+0.0f,+0.0f, +1.5f,+0.0f,+0.0f, +1.0f,+0.0f,+0.0f,
		+0.0f,+0.0f,+0.0f, +0.0f,+1.0f,+0.0f, +0.0f,+1.5f,+0.0f, +0.0f,+1.0f,+0.0f,
		+0.0f,+0.0f,+0.0f, +0.0f,+0.0f,+1.0f, +0.0f,+0.0f,+1.5f, +0.0f,+0.0f,+1.0f,
		// cube
		+0.0f,+1.0f,+0.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+0.0f, +1.0f,+1.0f,+1.0f,
		+0.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f,
		+0.0f,+1.0f,+0.0f, +1.0f,+1.0f,+1.0f, +0.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f,
		+1.0f,+1.0f,+0.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f,
		+1.0f,+0.0f,+0.0f, +1.0f,+1.0f,+1.0f, +1.0f,+0.0f,+1.0f, +1.0f,+1.0f,+1.0f,
		+0.0f,+0.0f,+1.0f, +1.0f,+1.0f,+1.0f, +1.0f,+0.0f,+1.0f, +1.0f,+1.0f,+1.0f,
		+1.0f,+0.0f,+0.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+0.0f, +1.0f,+1.0f,+1.0f,
		+0.0f,+0.0f,+1.0f, +1.0f,+1.0f,+1.0f, +0.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f,
		+1.0f,+0.0f,+1.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f, +1.0f,+1.0f,+1.0f
	};
	return std::make_unique<WireFrame>(verts);
}

int main(int argc, char** argv)
{
	// GL Initialization
	QApplication app(argc, argv);
	QOpenGLWidget qtGLWidget;
	{
		QSurfaceFormat surfaceFmt;
		surfaceFmt.setDepthBufferSize(24);
		surfaceFmt.setStencilBufferSize(8);
		surfaceFmt.setVersion(4, 5);
		surfaceFmt.setProfile(QSurfaceFormat::CoreProfile);
		qtGLWidget.setFormat(surfaceFmt);
	}
	qtGLWidget.show();
	assert(qtGLWidget.isValid());
	qtGLWidget.makeCurrent();

	assert(gladLoadGL() != 0);

	std::shared_ptr<VRContext> vrCtx;
	std::shared_ptr<StateModel> stateModel;
	std::unique_ptr<VREventHandler> vrInput;
	vrCtx = std::make_shared<VRContext>();
	stateModel = std::make_shared<StateModel>();
	try
	{
		vrInput = std::make_unique<VREventHandler>(
			std::string(PROJECT_SOURCE_DIR) + "/cfg/NeuroAnnoVRActions.json",
			vrCtx, stateModel);
	}
	catch (std::exception& e)
	{
		spdlog::critical("File: {0}, Line: {1}, Error: {2}", __FILE__,
			__LINE__, e.what());
		return 1;
	}

	struct
	{
		GLuint tex;
		CompVolumeRenderer::Subregion subrgn;
		std::shared_ptr<vs::CompVolume> volume;
		std::unique_ptr<kouek::CompVolumeFAVRRenderer> renderer;
	}volumeRender;
	try
	{
		{
			CompVolumeMonoEyeRenderer::CUDAParameter param;
			SetCUDACtx(0);
			param.ctx = GetCUDACtx();
			param.texUnitNum = 6;
			param.texUnitDim = { 1024,1024,1024 };
			volumeRender.renderer = CompVolumeFAVRRenderer::create(param);
		}

		kouek::VolumeConfig cfg(std::string(kouek::PROJECT_SOURCE_DIR) + "/cfg/VolumeCfg.json");
		volumeRender.volume =
			vs::CompVolume::Load(cfg.getResourcePath().c_str());
		volumeRender.volume->SetSpaceX(cfg.getSpaceX());
		volumeRender.volume->SetSpaceY(cfg.getSpaceY());
		volumeRender.volume->SetSpaceZ(cfg.getSpaceZ());
		//volumeRender.renderer->setVolume(volumeRender.volume);

		{
			vs::TransferFunc tf;
			tf.points.emplace_back(0, std::array<double, 4>{0.0, 0.1, 0.6, 0.0});
			tf.points.emplace_back(25, std::array<double, 4>{0.25, 0.5, 1.0, 0.0});
			tf.points.emplace_back(30, std::array<double, 4>{0.25, 0.5, 1.0, 0.2});
			tf.points.emplace_back(40, std::array<double, 4>{0.25, 0.5, 1.0, 0.1});
			tf.points.emplace_back(64, std::array<double, 4>{0.75, 0.50, 0.25, 0.4});
			tf.points.emplace_back(224, std::array<double, 4>{0.75, 0.75, 0.25, 0.6});
			tf.points.emplace_back(255, std::array<double, 4>{1.00, 0.75, 0.75, 0.4});
			volumeRender.renderer->setTransferFunc(tf);
		}
		{
			CompVolumeRenderer::LightParamter param;
			param.ka = 0.5f;
			param.kd = 0.8f;
			param.ks = 0.5f;
			param.shininess = 64.f;
			param.bkgrndColor = { .2f,.3f,.4f,.1f };
			volumeRender.renderer->setLightParam(param);
		}
		volumeRender.subrgn.center = { 3.24f,3.48f,5.21f };
		volumeRender.renderer->setStep(3000, cfg.getBaseSpace() * 0.3);
	}
	catch (std::exception& e)
	{
		spdlog::critical("File: {0}, Line: {1}, Error: {2}", __FILE__,
			__LINE__, e.what());
		return 1;
	}
	volumeRender.tex = createPlainTexture(2 * vrCtx->HMDRenderSizePerEye[0], vrCtx->HMDRenderSizePerEye[1]);

	GLint depthShaderMatrixPos, colorShaderMatrixPos, diffuseShaderMatrixPos;
	QOpenGLShaderProgram depthShader, colorShader, diffuseShader;
	// depthShader
	{
		const char* vertShaderCode =
			"#version 410 core\n"
			"uniform mat4 matrix;\n"
			"layout(location = 0) in vec3 position;\n"
			"layout(location = 1) in vec3 v3ColorIn;\n"
			"void main()\n"
			"{\n"
			"	gl_Position = matrix * vec4(position.xyz, 1.0);\n"
			"}\n";
		const char* fragShaderCode =
			"#version 410 core\n"
			"out vec4 outputColor;\n"
			"void main()\n"
			"{\n"
			"    outputColor = vec4(vec3(gl_FragCoord.z), 1.0);\n"
			"}\n";
		depthShader.addShaderFromSourceCode(
			QOpenGLShader::Vertex, vertShaderCode);
		depthShader.addShaderFromSourceCode(
			QOpenGLShader::Fragment, fragShaderCode);
		depthShader.link();
		assert(depthShader.isLinked());

		depthShaderMatrixPos = depthShader.uniformLocation("matrix");
		assert(depthShaderMatrixPos != -1);
	}
	{
		const char* vertShaderCode =
			"#version 410 core\n"
			"uniform mat4 matrix;\n"
			"layout(location = 0) in vec3 position;\n"
			"layout(location = 1) in vec3 v3ColorIn;\n"
			"out vec4 v4Color;\n"
			"void main()\n"
			"{\n"
			"	v4Color.xyz = v3ColorIn; v4Color.a = 1.0;\n"
			"	gl_Position = matrix * vec4(position.xyz, 1.0);\n"
			"}\n";
		const char* fragShaderCode =
			"#version 410 core\n"
			"in vec4 v4Color;\n"
			"out vec4 outputColor;\n"
			"void main()\n"
			"{\n"
			"    outputColor = v4Color;\n"
			"}\n";
		colorShader.addShaderFromSourceCode(
			QOpenGLShader::Vertex, vertShaderCode);
		colorShader.addShaderFromSourceCode(
			QOpenGLShader::Fragment, fragShaderCode);
		colorShader.link();
		assert(colorShader.isLinked());

		colorShaderMatrixPos = colorShader.uniformLocation("matrix");
		assert(colorShaderMatrixPos != -1);
	}
	{
		const char* vertShaderCode =
			"#version 410 core\n"
			"uniform mat4 matrix;\n"
			"layout(location = 0) in vec3 position;\n"
			"layout(location = 1) in vec3 v3NormalIn;\n"
			"layout(location = 2) in vec2 v2TexCoordsIn;\n"
			"out vec2 v2TexCoord;\n"
			"void main()\n"
			"{\n"
			"	v2TexCoord = v2TexCoordsIn;\n"
			"	gl_Position = matrix * vec4(position.xyz, 1);\n"
			"}\n";
		const char* fragShaderCode =
			"#version 410 core\n"
			"uniform sampler2D diffuse;\n"
			"in vec2 v2TexCoord;\n"
			"out vec4 outputColor;\n"
			"void main()\n"
			"{\n"
			"	outputColor = texture(diffuse, v2TexCoord);\n"
			"}\n";
		diffuseShader.addShaderFromSourceCode(
			QOpenGLShader::Vertex, vertShaderCode);
		diffuseShader.addShaderFromSourceCode(
			QOpenGLShader::Fragment, fragShaderCode);
		diffuseShader.link();
		assert(diffuseShader.isLinked());

		diffuseShaderMatrixPos = diffuseShader.uniformLocation("matrix");
		assert(diffuseShaderMatrixPos != -1);
	}

	struct
	{
		glm::mat4 transform = glm::identity<glm::mat4>();
		std::unique_ptr<WireFrame> model;
	}gizmo;
	gizmo.model = createGizmo();

	struct
	{
		GLuint VAO, VBO, EBO;
	}screenQuad;
	std::tie(screenQuad.VAO, screenQuad.VBO, screenQuad.EBO) = createScreenQuad();

	struct
	{
		GLuint FBO, colorTex, depthRBO;
	}depthFramebuffer2[2]{ 0 }, colorFramebuffer2[2]{ 0 }, submitFramebuffer2[2]{ 0 };
	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		std::tie(
			colorFramebuffer2[eyeIdx].FBO,
			colorFramebuffer2[eyeIdx].colorTex,
			colorFramebuffer2[eyeIdx].depthRBO) =
			createFrambuffer(vrCtx->HMDRenderSizePerEye[0], vrCtx->HMDRenderSizePerEye[1], true);
		std::tie(
			submitFramebuffer2[eyeIdx].FBO,
			submitFramebuffer2[eyeIdx].colorTex,
			submitFramebuffer2[eyeIdx].depthRBO) =
			createFrambuffer(vrCtx->HMDRenderSizePerEye[0], vrCtx->HMDRenderSizePerEye[1]);
		});

	/*volumeRender.renderer->registerGLResource(colorFramebuffer.colorTex, depthFramebuffer.depthRBO,
		vrCtx->HMDRenderSizePerEye[0], vrCtx->HMDRenderSizePerEye[1]);*/

	std::array<vr::Texture_t, 2> submitTex2 = {
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Left].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma},
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Right].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma}
	};

	Math::printGLMMat4(vrCtx->HMDToEye2[0], "HMDToEye L");
	Math::printGLMMat4(vrCtx->HMDToEye2[1], "HMDToEye R");
	Math::printGLMMat4(vrCtx->projection2[0], "Projection L");
	Math::printGLMMat4(vrCtx->projection2[1], "Projection R");
	
	while (!stateModel->shouldClose)
	{
		vrCtx->update();
		vrInput->update();

		if (!vrCtx->HMD->IsInputAvailable()) continue;

		std::array<glm::mat4, 2> MVP2;
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			MVP2[eyeIdx] = vrCtx->projection2[eyeIdx]
				* stateModel->camera.getViewMat(eyeIdx) * gizmo.transform;
			});
		static auto identity = glm::identity<glm::mat4>();
		GL_CHECK;
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		glEnable(GL_MULTISAMPLE);
		glClearColor(1.f, 1.f, 1.f, 1.f); // area without frag corresp to FarClip
		glClearColor(0.f, 0.f, 0.f, 1.f); // restore
		colorShader.bind();
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
			glViewport(0, 0, vrCtx->HMDRenderSizePerEye[0], vrCtx->HMDRenderSizePerEye[1]);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glUniformMatrix4fv(
				colorShaderMatrixPos, 1, GL_FALSE, (GLfloat*)&MVP2[eyeIdx]);
			gizmo.model->draw();
			});
		GL_CHECK;
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_MULTISAMPLE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, submitFramebuffer2[eyeIdx].FBO);
			glBlitFramebuffer(
				0, 0, vrCtx->HMDRenderSizePerEye[0], vrCtx->HMDRenderSizePerEye[1],
				0, 0, vrCtx->HMDRenderSizePerEye[0], vrCtx->HMDRenderSizePerEye[1],
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
			});
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		GL_CHECK;
		vr::EVRCompositorError err;
		err = vr::VRCompositor()->Submit(vr::Eye_Left, &submitTex2[vr::Eye_Left]);
		assert(err == vr::VRCompositorError_None);
		err = vr::VRCompositor()->Submit(vr::Eye_Right, &submitTex2[vr::Eye_Right]);
		assert(err == vr::VRCompositorError_None);
	}

	return 0;
}
