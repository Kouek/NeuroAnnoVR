#include <CMakeIn.h>
#include <renderer/Renderer.h>

#include <QtWidgets/qapplication.h>
#include <QtGUI/qopenglshaderprogram.h>

#include "VREventHandler.h"
#include "QtEventHandler.h"

#include <spdlog/spdlog.h>

using namespace kouek;


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

struct VolumeRenderType
{
	std::array<GLuint, 2> tex2;
	CompVolumeRenderer::Subregion subrgn;
	std::shared_ptr<vs::CompVolume> volume;
	std::unique_ptr<kouek::CompVolumeFAVRRenderer> renderer;
};
static void initVolumeRender(VolumeRenderType& volumeRender, std::shared_ptr<AppStates> states)
{
	{
		CompVolumeMonoEyeRenderer::CUDAParameter param;
		SetCUDACtx(0);
		param.ctx = GetCUDACtx();
		param.texUnitNum = 1;
		param.texUnitDim = { 1024,1024,1024 };
		volumeRender.renderer = CompVolumeFAVRRenderer::create(param);
	}
	try
	{
		kouek::VolumeConfig cfg(std::string(kouek::PROJECT_SOURCE_DIR) + "/cfg/VolumeCfg.json");
		volumeRender.volume =
			vs::CompVolume::Load(cfg.getResourcePath().c_str());
		volumeRender.volume->SetSpaceX(cfg.getSpaceX());
		volumeRender.volume->SetSpaceY(cfg.getSpaceY());
		volumeRender.volume->SetSpaceZ(cfg.getSpaceZ());
		volumeRender.renderer->setStep(3000, cfg.getBaseSpace() * 0.3);
		volumeRender.renderer->setVolume(volumeRender.volume);
	}
	catch (std::exception& e)
	{
		spdlog::critical("File: {0}, Line: {1}, Error: {2}", __FILE__,
			__LINE__, e.what());
		return;
	}
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
	volumeRender.subrgn.halfW = volumeRender.subrgn.halfH =
		volumeRender.subrgn.halfD = .08f;
	volumeRender.subrgn.rotation = volumeRender.subrgn.fromWorldToSubrgn =
		glm::identity<glm::mat4>();
	volumeRender.tex2[0] = createPlainTexture(states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
	volumeRender.tex2[1] = createPlainTexture(states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
}

struct ShaderProgramsType
{
	GLint depthShaderMatrixPos, colorShaderMatrixPos, diffuseShaderMatrixPos;
	QOpenGLShaderProgram depthShader, colorShader, diffuseShader;
};
static void initShaderPrograms(ShaderProgramsType& shaders)
{
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
		shaders.depthShader.addShaderFromSourceCode(
			QOpenGLShader::Vertex, vertShaderCode);
		shaders.depthShader.addShaderFromSourceCode(
			QOpenGLShader::Fragment, fragShaderCode);
		shaders.depthShader.link();
		assert(shaders.depthShader.isLinked());

		shaders.depthShaderMatrixPos = shaders.depthShader.uniformLocation("matrix");
		assert(shaders.depthShaderMatrixPos != -1);
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
		shaders.colorShader.addShaderFromSourceCode(
			QOpenGLShader::Vertex, vertShaderCode);
		shaders.colorShader.addShaderFromSourceCode(
			QOpenGLShader::Fragment, fragShaderCode);
		shaders.colorShader.link();
		assert(shaders.colorShader.isLinked());

		shaders.colorShaderMatrixPos = shaders.colorShader.uniformLocation("matrix");
		assert(shaders.colorShaderMatrixPos != -1);
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
		shaders.diffuseShader.addShaderFromSourceCode(
			QOpenGLShader::Vertex, vertShaderCode);
		shaders.diffuseShader.addShaderFromSourceCode(
			QOpenGLShader::Fragment, fragShaderCode);
		shaders.diffuseShader.link();
		assert(shaders.diffuseShader.isLinked());

		shaders.diffuseShaderMatrixPos = shaders.diffuseShader.uniformLocation("matrix");
		assert(shaders.diffuseShaderMatrixPos != -1);
	}
}

int main(int argc, char** argv)
{
	// init Qt and GL Context (by EditorWindow.QOpenGLWidget) ====>
	QApplication qtApp(argc, argv);
	EditorWindow editorWindow;
	editorWindow.showMaximized();
	editorWindow.getVRView()->makeCurrent();
	// init States and Event Handler (both VR and Qt) ====>
	std::shared_ptr<AppStates> states = std::make_shared<AppStates>();
	std::unique_ptr<EventHandler> vrEvntHndler;
	try
	{
		vrEvntHndler = std::make_unique<VREventHandler>(
			std::string(PROJECT_SOURCE_DIR) + "/cfg/NeuroAnnoVRActions.json",
			states);
	}
	catch (std::exception& e)
	{
		states->canVRRun = false;
		spdlog::error("File: {0}, Line: {1}, Error: {2}", __FILE__,
			__LINE__, e.what());
	}
	std::unique_ptr<EventHandler> qtEvntHndler;
	qtEvntHndler = std::make_unique<QtEventHandler>(&editorWindow, states);

	// init Volume and GL Resource ====>
	VolumeRenderType volumeRender;
	initVolumeRender(volumeRender, states);

	ShaderProgramsType shaders;
	initShaderPrograms(shaders);

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
			depthFramebuffer2[eyeIdx].FBO,
			depthFramebuffer2[eyeIdx].colorTex,
			depthFramebuffer2[eyeIdx].depthRBO) =
			createFrambuffer(states->HMDRenderSizePerEye[0],
				states->HMDRenderSizePerEye[1]);
		std::tie(
			colorFramebuffer2[eyeIdx].FBO,
			colorFramebuffer2[eyeIdx].colorTex,
			colorFramebuffer2[eyeIdx].depthRBO) =
			createFrambuffer(states->HMDRenderSizePerEye[0],
				states->HMDRenderSizePerEye[1], true);
		std::tie(
			submitFramebuffer2[eyeIdx].FBO,
			submitFramebuffer2[eyeIdx].colorTex,
			submitFramebuffer2[eyeIdx].depthRBO) =
			createFrambuffer(states->HMDRenderSizePerEye[0],
				states->HMDRenderSizePerEye[1]);
		});

	volumeRender.renderer->registerGLResource(
		volumeRender.tex2[vr::Eye_Left], volumeRender.tex2[vr::Eye_Right],
		depthFramebuffer2[vr::Eye_Left].colorTex, depthFramebuffer2[vr::Eye_Right].colorTex,
		states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
	editorWindow.getVRView()->setInputFBOs(submitFramebuffer2[vr::Eye_Left].FBO,
		submitFramebuffer2[vr::Eye_Right].FBO, states->HMDRenderSizePerEye);
	std::array<vr::Texture_t, 2> submitVRTex2 = {
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Left].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma},
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Right].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma}
	};

	Math::printGLMMat4(states->HMDToEye2[vr::Eye_Left], "HMDToEye L");
	Math::printGLMMat4(states->HMDToEye2[vr::Eye_Right], "HMDToEye R");
	Math::printGLMMat4(states->projection2[vr::Eye_Left], "Projection L");
	Math::printGLMMat4(states->projection2[vr::Eye_Right], "Projection R");

	states->camera.setHeadPos(
		glm::vec3(volumeRender.subrgn.halfW * 2.f,
			volumeRender.subrgn.halfW * 2.f,
			volumeRender.subrgn.halfW * 2.f));
	std::array<glm::mat4, 2> MVP2;
	while (states->canRun)
	{
		qtApp.processEvents();
		editorWindow.getVRView()->makeCurrent();
		if (states->canVRRun)
			vrEvntHndler->update();
		
		gizmo.transform = glm::scale(glm::identity<glm::mat4>(),
			glm::vec3(volumeRender.subrgn.halfW * 2.f,
				volumeRender.subrgn.halfW * 2.f,
				volumeRender.subrgn.halfW * 2.f));
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			MVP2[eyeIdx] = states->projection2[eyeIdx]
				* states->camera.getViewMat(eyeIdx) * gizmo.transform;
			});
		static auto identity = glm::identity<glm::mat4>();
		
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		glEnable(GL_MULTISAMPLE);
		glClearColor(1.f, 1.f, 1.f, 1.f); // area without frag corresp to FarClip
		shaders.depthShader.bind();
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			glBindFramebuffer(GL_FRAMEBUFFER, depthFramebuffer2[eyeIdx].FBO);
			glViewport(0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glUniformMatrix4fv(
				shaders.colorShaderMatrixPos, 1, GL_FALSE, (GLfloat*)&MVP2[eyeIdx]);
			gizmo.model->draw();
			});

		glClearColor(0.f, 0.f, 0.f, 1.f); // restore
		shaders.colorShader.bind();
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
			glViewport(0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glUniformMatrix4fv(
				shaders.colorShaderMatrixPos, 1, GL_FALSE, (GLfloat*)&MVP2[eyeIdx]);
			gizmo.model->draw();
			});
		
		{
			auto [right, forward, up, lftEyePos, rhtEyePos] =
				states->camera.getRFUP2();
			glm::mat4 rotation(
				right.x, right.y, right.z, 0,
				up.x, up.y, up.z, 0,
				-forward.x, -forward.y, -forward.z, 0,
				0, 0, 0, 1.f);
			volumeRender.renderer->setCamera(
				{ lftEyePos, rhtEyePos, rotation,
				states->unProjection2[0],
				states->nearClip, states->farClip });
		}
		volumeRender.renderer->setSubregion(volumeRender.subrgn);
		volumeRender.renderer->render();

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		shaders.diffuseShader.bind();
		glUniformMatrix4fv(
			shaders.diffuseShaderMatrixPos, 1, GL_FALSE,
			(GLfloat*)&identity);
		glBindVertexArray(screenQuad.VAO);
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
			{
				glBindTexture(GL_TEXTURE_2D, volumeRender.tex2[eyeIdx]);
				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void*)0);
				glBindTexture(GL_TEXTURE_2D, 0);
			}
			});
		glBindVertexArray(0);

		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, submitFramebuffer2[eyeIdx].FBO);
			glBlitFramebuffer(
				0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1],
				0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1],
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
			});
		
		editorWindow.getVRView()->update();
		if (states->canVRRun)
		{
			vr::VRCompositor()->Submit(vr::Eye_Left, &submitVRTex2[vr::Eye_Left]);
			vr::VRCompositor()->Submit(vr::Eye_Right, &submitVRTex2[vr::Eye_Right]);
		}
	}

	qtApp.exit();
	return 0;
}
