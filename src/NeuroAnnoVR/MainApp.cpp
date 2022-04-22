#include "MainApp.h"

#include <spdlog/spdlog.h>

#include <util/VolumeCfg.h>

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

static std::tuple<GLuint, GLuint, GLuint> createFrambuffer(
	uint32_t w, uint32_t h, bool useMultiSample = false)
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

static std::tuple<GLuint, GLuint, GLuint> createPlane()
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

static std::unique_ptr<WireFrame> createAnnotationBall()
{
	std::vector<GLfloat> verts = {
		// cube
		-0.5f,-0.5f,-0.5f, +1.0f,+1.0f,+1.0f, +0.5f,-0.5f,-0.5f, +1.0f,+1.0f,+1.0f,
		-0.5f,-0.5f,-0.5f, +1.0f,+1.0f,+1.0f, -0.5f,+0.5f,-0.5f, +1.0f,+1.0f,+1.0f,
		-0.5f,-0.5f,-0.5f, +1.0f,+1.0f,+1.0f, -0.5f,-0.5f,+0.5f, +1.0f,+1.0f,+1.0f,
		-0.5f,+0.5f,+0.5f, +1.0f,+1.0f,+1.0f, -0.5f,+0.5f,-0.5f, +1.0f,+1.0f,+1.0f,
		-0.5f,+0.5f,+0.5f, +1.0f,+1.0f,+1.0f, -0.5f,-0.5f,+0.5f, +1.0f,+1.0f,+1.0f,
		-0.5f,+0.5f,+0.5f, +1.0f,+1.0f,+1.0f, +0.5f,+0.5f,+0.5f, +1.0f,+1.0f,+1.0f,
		+0.5f,-0.5f,+0.5f, +1.0f,+1.0f,+1.0f, +0.5f,-0.5f,-0.5f, +1.0f,+1.0f,+1.0f,
		+0.5f,-0.5f,+0.5f, +1.0f,+1.0f,+1.0f, +0.5f,+0.5f,+0.5f, +1.0f,+1.0f,+1.0f,
		+0.5f,-0.5f,+0.5f, +1.0f,+1.0f,+1.0f, -0.5f,-0.5f,+0.5f, +1.0f,+1.0f,+1.0f,
		+0.5f,+0.5f,-0.5f, +1.0f,+1.0f,+1.0f, +0.5f,+0.5f,+0.5f, +1.0f,+1.0f,+1.0f,
		+0.5f,+0.5f,-0.5f, +1.0f,+1.0f,+1.0f, +0.5f,-0.5f,-0.5f, +1.0f,+1.0f,+1.0f,
		+0.5f,+0.5f,-0.5f, +1.0f,+1.0f,+1.0f, -0.5f,+0.5f,-0.5f, +1.0f,+1.0f,+1.0f
	};
	return std::make_unique<WireFrame>(verts);
}

static void initVolumeRender(
	VolumeRenderType& volumeRender, std::shared_ptr<AppStates> states)
{
	{
		CompVolumeMonoEyeRenderer::CUDAParameter param;
		SetCUDACtx(0);
		param.ctx = GetCUDACtx();
		param.texUnitNum = 6;
		param.texUnitDim = { 1024,1024,1024 };
		volumeRender.renderer = CompVolumeFAVRRenderer::create(param);
	}
	try
	{
		kouek::VolumeConfig cfg(std::string(kouek::PROJECT_SOURCE_DIR)
			+ "/cfg/VolumeCfg.json");
		volumeRender.volume =
			vs::CompVolume::Load(cfg.getResourcePath().c_str());
		const float scale = 5.f;
		volumeRender.volume->SetSpaceX(cfg.getSpaceX() * scale);
		volumeRender.volume->SetSpaceY(cfg.getSpaceY() * scale);
		volumeRender.volume->SetSpaceZ(cfg.getSpaceZ() * scale);
		volumeRender.renderer->setStep(1024, cfg.getBaseSpace() * scale * .3f );
		AppStates::subrgnMoveSensity = AppStates::moveSensity = std::max(
			volumeRender.volume->GetVolumeSpaceX(),
			volumeRender.volume->GetVolumeSpaceY());
		volumeRender.renderer->setVolume(volumeRender.volume);
		volumeRender.renderer->setTransferFunc(cfg.getTF());
	}
	catch (std::exception& e)
	{
		spdlog::critical("File: {0}, Line: {1}, Error: {2}", __FILE__,
			__LINE__, e.what());
		return;
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
	{
		const auto& blkLenInfo = volumeRender.volume->GetBlockLength();
		volumeRender.noPadBlkLen = blkLenInfo[0] - blkLenInfo[1] * 2;
		states->subrgn.center = { 22.0718937f, 36.6882820f, 32.2800331f };
		states->subrgn.halfW = volumeRender.noPadBlkLen / 2 * volumeRender.volume->GetVolumeSpaceX();
		states->subrgn.halfH = volumeRender.noPadBlkLen / 2 * volumeRender.volume->GetVolumeSpaceY();
		states->subrgn.halfD = volumeRender.noPadBlkLen / 16 * volumeRender.volume->GetVolumeSpaceZ();
		states->subrgn.rotation = states->subrgn.fromWorldToSubrgn =
			glm::identity<glm::mat4>();
	}
	volumeRender.tex2[0] = createPlainTexture(states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
	volumeRender.tex2[1] = createPlainTexture(states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
}

kouek::MainApp::MainApp(int argc, char** argv)
{
	// init Qt and GL Context (by EditorWindow.QOpenGLWidget) ====>
	qtApp = std::make_unique<QApplication>(argc, argv);
	editorWindow = std::make_unique<EditorWindow>();
	editorWindow->showMaximized();
	editorWindow->getVRView()->makeCurrent(); // set it as default GL Context

	// init States and Event Handler (both VR and Qt) ====>
	states = std::make_shared<AppStates>();
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
	qtEvntHndler = std::make_unique<QtEventHandler>(
		editorWindow.get(), states);

	// init Volume and GL Resource ====>

	shaders = std::make_unique<Shaders>();
	
	initVolumeRender(volumeRender, states);
	states->renderer = volumeRender.renderer.get();

	pathRenderer = std::make_unique<GLPathRenderer>();
	GLPathRenderer::rootVertSize = 20.f;
	GLPathRenderer::endVertSize = 5.f;
	GLPathRenderer::selectedVertSize = 8.f;
	GLPathRenderer::selectVertSize = 3.f;
	GLPathRenderer::lineWidth = 2.f;
	states->pathRenderer = pathRenderer.get();
	volumeRender.renderer->setInteractionParam(states->game.intrctParam);

	gizmo.model = createGizmo();

	annoBall.model = createAnnotationBall();

	intrctPoint.model = std::make_unique<Point>();
	intrctPoint.model->setColorData(GLPathRenderer::selectedVertColor);

	std::tie(screenQuad.VAO, screenQuad.VBO, screenQuad.EBO) = createPlane();
	std::tie(handUIQuad[0].VAO, handUIQuad[0].VBO, handUIQuad[0].EBO) = createPlane();
	std::tie(handUIQuad[1].VAO, handUIQuad[1].VBO, handUIQuad[1].EBO) = createPlane();

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
	std::tie(
		pathSelectFramebuffer.FBO,
		pathSelectFramebuffer.colorTex,
		pathSelectFramebuffer.depthRBO) =
		createFrambuffer(states->HMDRenderSizePerEye[0],
			states->HMDRenderSizePerEye[1]);

	volumeRender.renderer->registerGLResource(
		volumeRender.tex2[vr::Eye_Left], volumeRender.tex2[vr::Eye_Right],
		depthFramebuffer2[vr::Eye_Left].colorTex, depthFramebuffer2[vr::Eye_Right].colorTex,
		states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
	editorWindow->getVRView()->setInputFBOs(submitFramebuffer2[vr::Eye_Left].FBO,
		submitFramebuffer2[vr::Eye_Right].FBO, states->HMDRenderSizePerEye);

	{
		states->camera.setHeadPos(
			glm::vec3(states->subrgn.halfW,
				states->subrgn.halfH,
				states->subrgn.halfD * 4.f));

		gizmo.transform = glm::scale(glm::identity<glm::mat4>(),
			glm::vec3(states->subrgn.halfW * 2,
				states->subrgn.halfH * 2,
				states->subrgn.halfD * 2));
		auto invTranslation = glm::translate(glm::identity<glm::mat4>(),
			glm::vec3(-states->subrgn.halfW,
				-states->subrgn.halfH,
				-states->subrgn.halfD));
		auto translation = glm::translate(glm::identity<glm::mat4>(),
			glm::vec3(states->subrgn.halfW,
				states->subrgn.halfH,
				states->subrgn.halfD));
		auto TRInvT = translation * states->subrgn.rotation
			* invTranslation;
		gizmo.transform = TRInvT * gizmo.transform;

		states->subrgn.fromWorldToSubrgn =
			Math::inversePose(TRInvT);

		volumeRender.renderer->setSubregion(states->subrgn);
	}

	Math::printGLMMat4(states->eyeToHMD2[vr::Eye_Left], "HMDToEye L");
	Math::printGLMMat4(states->eyeToHMD2[vr::Eye_Right], "HMDToEye R");
	Math::printGLMMat4(states->projection2[vr::Eye_Left], "Projection L");
	Math::printGLMMat4(states->projection2[vr::Eye_Right], "Projection R");
	printf("Render %d x %d pixels per Eye\n", states->HMDRenderSizePerEye[0],
		states->HMDRenderSizePerEye[1]);
}

kouek::MainApp::~MainApp() {}

int kouek::MainApp::run()
{
	if (!states->canRun) return 1;

	std::array<vr::Texture_t, 2> submitVRTex2 = {
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Left].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma},
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Right].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma}
	};

	while (states->canRun)
	{
		qtApp->processEvents();
		editorWindow->getVRView()->makeCurrent();

		if (states->canVRRun)
			vrEvntHndler->update();
		qtEvntHndler->update();

		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			VP2[eyeIdx] = states->projection2[eyeIdx]
				* states->camera.getViewMat(eyeIdx);
			});
		if (states->showHandUI2[VRContext::Hand_Left]
			|| states->showHandUI2[VRContext::Hand_Right])
			drawUI();
		else
			drawScene();

		editorWindow->getVRView()->update();
		if (states->canVRRun)
		{
			vr::VRCompositor()->Submit(vr::Eye_Left, &submitVRTex2[vr::Eye_Left]);
			vr::VRCompositor()->Submit(vr::Eye_Right, &submitVRTex2[vr::Eye_Right]);
		}
	}

	qtApp->exit();
	return 0;
}

void kouek::MainApp::drawUI()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	glClearColor(0.f, 0.f, 0.f, 1.f);

	shaders->diffuseShader.program.bind();
	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		handUIMVP2[eyeIdx] = glm::translate(
			glm::identity<glm::mat4>(), states->UITranslate);
		handUIMVP2[eyeIdx] = VP2[eyeIdx] * handUIMVP2[eyeIdx];

		glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
		glViewport(0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		VRContext::forHandsDo([&](uint8_t hndIdx) {
			if (states->hand2[hndIdx].show)
			{
				handMVP22[eyeIdx][hndIdx] = VP2[eyeIdx]
					* states->hand2[hndIdx].transform;

				glUniformMatrix4fv(
					shaders->diffuseShader.matPos, 1, GL_FALSE,
					(GLfloat*)&handMVP22[eyeIdx][hndIdx]);
				states->hand2[hndIdx].model->draw();
			}
			if (states->showHandUI2[hndIdx])
			{
				glUniformMatrix4fv(
					shaders->diffuseShader.matPos, 1, GL_FALSE,
					(GLfloat*)&handUIMVP2[hndIdx]);
				glBindVertexArray(handUIQuad[hndIdx].VAO);
				glBindTexture(GL_TEXTURE_2D, qtEvntHndler->getHandUITex(hndIdx));
				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void*)0);
				glBindVertexArray(0);
			}
			});
		});

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, submitFramebuffer2[eyeIdx].FBO);
		glBlitFramebuffer(
			0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1],
			0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1],
			GL_COLOR_BUFFER_BIT, GL_LINEAR);
		});
}

void kouek::MainApp::drawScene()
{
	static auto identity = glm::identity<glm::mat4>();
	if (states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].show)
	{
		auto& handZ = states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].transform[2];
		auto& handPos = states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].transform[3];
		glm::vec3 ballPos = handPos - ANNO_BALL_DIST_FROM_HAND * handZ;
		states->game.intrctParam.dat.ball.startPos.x = ballPos.x - ANNO_BALL_RADIUS;
		states->game.intrctParam.dat.ball.startPos.y = ballPos.y - ANNO_BALL_RADIUS;
		states->game.intrctParam.dat.ball.startPos.z = ballPos.z - ANNO_BALL_RADIUS;
		anno.ball.transform = glm::translate(glm::identity<glm::mat4>(), ballPos);
		anno.ball.transform = glm::scale(anno.ball.transform, glm::vec3{ ANNO_BALL_DIAMETER });

		anno.ball.projectedPos = VP2[0] * glm::vec4{ ballPos,1.f };
		anno.ball.projectedPos /= anno.ball.projectedPos.w;

		anno.ball.screenPos.x = .5f * (anno.ball.projectedPos.x + 1.f);
		anno.ball.screenPos.y = .5f * (anno.ball.projectedPos.y + 1.f);
		if (states->game.shouldSelectVertex
			&& anno.ball.screenPos.x >= 0 && anno.ball.screenPos.x < 1.f
			&& anno.ball.screenPos.y >= 0 && anno.ball.screenPos.y < 1.f)
		{
			std::array<GLubyte, 4> id4;
			glBindFramebuffer(GL_FRAMEBUFFER, pathSelectFramebuffer.FBO);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glReadPixels(
				(GLint)floor(states->HMDRenderSizePerEye[0] * anno.ball.screenPos.x),
				(GLint)floor(states->HMDRenderSizePerEye[1] * anno.ball.screenPos.y),
				1, 1, GL_RGBA, GL_UNSIGNED_BYTE, id4.data());
			GLuint vertID = (GLuint)id4[0]
				| ((GLuint)id4[1] << 8)
				| ((GLuint)id4[2] << 16)
				| ((GLuint)id4[3] << 24);
			if (vertID != std::numeric_limits<GLuint>::max())
				pathRenderer->startVertex(vertID);
		}
	}
	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		gizmoMVP2[eyeIdx] = VP2[eyeIdx] * gizmo.transform;
		anno.ball.MVP2[eyeIdx] = VP2[eyeIdx] * anno.ball.transform;
		VRContext::forHandsDo([&](uint8_t hndIdx) {
			handMVP22[eyeIdx][hndIdx] = VP2[eyeIdx]
				* states->hand2[hndIdx].transform;
			});
		});

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glClearColor(1.f, 1.f, 1.f, 1.f); // area without frag corresp to FarClip
	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		glBindFramebuffer(GL_FRAMEBUFFER, depthFramebuffer2[eyeIdx].FBO);
		glViewport(0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shaders->depthShader.program.bind();
		if (states->showGizmo)
		{
			glUniformMatrix4fv(
				shaders->depthShader.matPos, 1, GL_FALSE, (GLfloat*)&gizmoMVP2[eyeIdx]);
			gizmo.model->draw();
		}

		if (states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].show)
		{
			glUniformMatrix4fv(
				shaders->depthShader.matPos, 1, GL_FALSE,
				(GLfloat*)&anno.ball.MVP2[eyeIdx]);
			annoBall.model->draw();
		}

		VRContext::forHandsDo([&](uint8_t hndIdx) {
			if (!states->hand2[hndIdx].show) return;
			glUniformMatrix4fv(
				shaders->depthShader.matPos, 1, GL_FALSE,
				(GLfloat*)&handMVP22[eyeIdx][hndIdx]);
			states->hand2[hndIdx].model->draw();
			});

		if (static_cast<uint32_t>(states->game.intrctActMode) & (
			static_cast<uint32_t>(InteractionActionMode::AddPath)
			| static_cast<uint32_t>(InteractionActionMode::AddVertex)))
		{
			shaders->zeroDepthShader.program.bind();
			glUniformMatrix4fv(
				shaders->colorShader.matPos, 1, GL_FALSE, (GLfloat*)&VP2[eyeIdx]);
			if (states->game.intrctActMode == InteractionActionMode::AddPath)
				glPointSize(GLPathRenderer::rootVertSize);
			else
				glPointSize(GLPathRenderer::endVertSize);
			intrctPoint.model->draw();
		}
		else if (eyeIdx == vr::Eye_Left)
		{
			shaders->zeroDepthShader.program.bind();
			glUniformMatrix4fv(
				shaders->colorShader.matPos, 1, GL_FALSE, (GLfloat*)&identity);
			glPointSize(GLPathRenderer::selectVertSize);
			intrctPoint.model->draw();
		}

		shaders->pathDepthShader.program.bind();
		glUniformMatrix4fv(
			shaders->pathDepthShader.matPos, 1, GL_FALSE, (GLfloat*)&VP2[eyeIdx]);
		pathRenderer->draw();
		});

	if (states->game.intrctActMode == InteractionActionMode::SelectVertex)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, pathSelectFramebuffer.FBO);
		glViewport(0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		shaders->pathSelectShader.program.bind();
		glUniformMatrix4fv(
			shaders->pathSelectShader.matPos, 1, GL_FALSE, (GLfloat*)&VP2[0]);
		pathRenderer->drawVertIDs();
	}

	glEnable(GL_MULTISAMPLE);
	glClearColor(0.f, 0.f, 0.f, 1.f); // restore
	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
		glViewport(0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shaders->colorShader.program.bind();
		if (states->showGizmo)
		{
			glUniformMatrix4fv(
				shaders->colorShader.matPos, 1, GL_FALSE, (GLfloat*)&gizmoMVP2[eyeIdx]);
			gizmo.model->draw();
		}

		if (states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].show)
		{
			glUniformMatrix4fv(
				shaders->colorShader.matPos, 1, GL_FALSE,
				(GLfloat*)&anno.ball.MVP2[eyeIdx]);
			annoBall.model->draw();
		}

		if (static_cast<uint32_t>(states->game.intrctActMode) & (
			static_cast<uint32_t>(InteractionActionMode::AddPath)
			| static_cast<uint32_t>(InteractionActionMode::AddVertex)))
		{
			glUniformMatrix4fv(
				shaders->colorShader.matPos, 1, GL_FALSE, (GLfloat*)&VP2[eyeIdx]);
			if (states->game.intrctActMode == InteractionActionMode::AddPath)
				glPointSize(GLPathRenderer::rootVertSize);
			else
				glPointSize(GLPathRenderer::endVertSize);
			intrctPoint.model->draw();
		}
		else if (eyeIdx == vr::Eye_Left)
		{
			glUniformMatrix4fv(
				shaders->colorShader.matPos, 1, GL_FALSE, (GLfloat*)&identity);
			glPointSize(GLPathRenderer::selectVertSize);
			intrctPoint.model->draw();
		}

		VRContext::forHandsDo([&](uint8_t hndIdx) {
			if (!states->hand2[hndIdx].show) return;
			shaders->diffuseShader.program.bind();
			glUniformMatrix4fv(
				shaders->diffuseShader.matPos, 1, GL_FALSE,
				(GLfloat*)&handMVP22[eyeIdx][hndIdx]);
			states->hand2[hndIdx].model->draw();
			});

		shaders->pathColorShader.program.bind();
		glUniformMatrix4fv(
			shaders->pathColorShader.matPos, 1, GL_FALSE, (GLfloat*)&VP2[eyeIdx]);
		pathRenderer->draw(shaders->pathColorShader.colorPos);
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
			states->unProjection2[vr::Eye_Left],states->unProjection2[vr::Eye_Right],
			states->nearClip, states->farClip });
	}
	if (static_cast<uint32_t>(states->game.intrctActMode) & (
		static_cast<uint32_t>(InteractionActionMode::AddPath)
		| static_cast<uint32_t>(InteractionActionMode::AddVertex)))
	{
		volumeRender.renderer->setInteractionParam(states->game.intrctParam);
		volumeRender.renderer->render(&states->game.intrctPos, states->renderTar);
		intrctPoint.model->setPosData(states->game.intrctPos);
	}
	else
	{
		volumeRender.renderer->render(nullptr, states->renderTar);
		intrctPoint.model->setPosData(glm::vec3(anno.ball.projectedPos));
	}

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
	shaders->diffuseShader.program.bind();
	glUniformMatrix4fv(
		shaders->diffuseShader.matPos, 1, GL_FALSE, (GLfloat*)&identity);
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

	glDisable(GL_MULTISAMPLE);
	VRContext::forEyesDo([&](uint8_t eyeIdx) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, submitFramebuffer2[eyeIdx].FBO);
		glBlitFramebuffer(
			0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1],
			0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1],
			GL_COLOR_BUFFER_BIT, GL_LINEAR);
		});
}
