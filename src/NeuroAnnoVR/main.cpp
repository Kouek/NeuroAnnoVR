#include <CMakeIn.h>
#include <renderer/Renderer.h>

#include <QtWidgets/qapplication.h>

#include "VREventHandler.h"
#include "QtEventHandler.h"
#include "Shaders.h"

#include <spdlog/spdlog.h>

#include <util/VolumeCfg.h>
#include <util/RenderObj.h>

using namespace kouek;

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

static std::tuple<GLuint, GLuint> createPoint()
{
	GLuint VAO, VBO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	{
		GLfloat verts[] = { 0.f, 0.f, 0.f,	1.f, .5f, 1.f };
		glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const void*)(0));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const void*)(sizeof(GLfloat) * 3));
	}
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return { VAO,VBO };
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

struct VolumeRenderType
{
	uint32_t noPadBlkLen;
	std::array<GLuint, 2> tex2;
	std::shared_ptr<vs::CompVolume> volume;
};
static void initVolumeRender(VolumeRenderType& volumeRender, std::shared_ptr<AppStates> states)
{
	{
		CompVolumeMonoEyeRenderer::CUDAParameter param;
		SetCUDACtx(0);
		param.ctx = GetCUDACtx();
		param.texUnitNum = 6;
		param.texUnitDim = { 1024,1024,1024 };
		states->renderer = CompVolumeFAVRRenderer::create(param);
	}
	try
	{
		kouek::VolumeConfig cfg(std::string(kouek::PROJECT_SOURCE_DIR) + "/cfg/VolumeCfg.json");
		volumeRender.volume =
			vs::CompVolume::Load(cfg.getResourcePath().c_str());
		const float scale = 10.f;
		volumeRender.volume->SetSpaceX(cfg.getSpaceX() * scale);
		volumeRender.volume->SetSpaceY(cfg.getSpaceY() * scale);
		volumeRender.volume->SetSpaceZ(cfg.getSpaceZ() * scale);
		states->renderer->setStep(3000, cfg.getBaseSpace() * scale * .3f );
		AppStates::subrgnMoveSensity = AppStates::moveSensity = std::max(
			volumeRender.volume->GetVolumeSpaceX(),
			volumeRender.volume->GetVolumeSpaceY());
		states->renderer->setVolume(volumeRender.volume);
		states->renderer->setTransferFunc(cfg.getTF());
	}
	catch (std::exception& e)
	{
		spdlog::critical("File: {0}, Line: {1}, Error: {2}", __FILE__,
			__LINE__, e.what());
		return;
	}
	//{
	//	vs::TransferFunc tf;
	//	tf.points.emplace_back(0, std::array<double, 4>{0.0, 0.1, 0.6, 0.0});
	//	tf.points.emplace_back(25, std::array<double, 4>{0.25, 0.5, 1.0, 0.0});
	//	tf.points.emplace_back(30, std::array<double, 4>{0.25, 0.5, 1.0, 0.2});
	//	tf.points.emplace_back(40, std::array<double, 4>{0.25, 0.5, 1.0, 0.1});
	//	tf.points.emplace_back(64, std::array<double, 4>{0.75, 0.50, 0.25, 0.4});
	//	tf.points.emplace_back(224, std::array<double, 4>{0.75, 0.75, 0.25, 0.6});
	//	tf.points.emplace_back(255, std::array<double, 4>{1.00, 0.75, 0.75, 0.4});
	//	states->renderer->setTransferFunc(tf);
	//}
	{
		CompVolumeRenderer::LightParamter param;
		param.ka = 0.5f;
		param.kd = 0.8f;
		param.ks = 0.5f;
		param.shininess = 64.f;
		param.bkgrndColor = { .2f,.3f,.4f,.1f };
		states->renderer->setLightParam(param);
	}
	{
		const auto& blkLenInfo = volumeRender.volume->GetBlockLength();
		volumeRender.noPadBlkLen = blkLenInfo[0] - blkLenInfo[1] * 2;
		states->subrgn.center = { 44.4351540f, 73.9647980f, 62.69787188f };
		states->subrgn.halfW = volumeRender.noPadBlkLen / 4 * volumeRender.volume->GetVolumeSpaceX();
		states->subrgn.halfH = volumeRender.noPadBlkLen / 4 * volumeRender.volume->GetVolumeSpaceY();
		states->subrgn.halfD = volumeRender.noPadBlkLen / 16 * volumeRender.volume->GetVolumeSpaceZ();
		states->subrgn.rotation = states->subrgn.fromWorldToSubrgn =
			glm::identity<glm::mat4>();
	}
	volumeRender.tex2[0] = createPlainTexture(states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
	volumeRender.tex2[1] = createPlainTexture(states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
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

	Shaders shaders;

	struct
	{
		glm::mat4 transform = glm::identity<glm::mat4>();
		std::unique_ptr<WireFrame> model;
	}gizmo;
	gizmo.model = createGizmo();

	constexpr float ANNO_BALL_RADIUS = .02f;
	constexpr float ANNO_BALL_DIAMETER = 2 * ANNO_BALL_RADIUS;
	constexpr float ANNO_BALL_DIST_FROM_HAND = ANNO_BALL_DIAMETER * 2;
	struct
	{
		glm::mat4 initTr = glm::scale(glm::identity<glm::mat4>(), glm::vec3{ ANNO_BALL_DIAMETER });
		glm::mat4 transform;
		std::unique_ptr<WireFrame> model;
	}annotationBall;
	annotationBall.model = createAnnotationBall();
	CompVolumeFAVRRenderer::InteractionParameter intrctParam;
	intrctParam.mode = CompVolumeFAVRRenderer::InteractionMode::AnnotationBall;
	intrctParam.dat.ball.AABBSize = glm::vec3{ ANNO_BALL_DIAMETER };

	struct
	{
		GLuint VAO, VBO;
	}intrctPoint;
	std::tie(intrctPoint.VAO, intrctPoint.VBO) = createPoint();

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

	states->renderer->registerGLResource(
		volumeRender.tex2[vr::Eye_Left], volumeRender.tex2[vr::Eye_Right],
		depthFramebuffer2[vr::Eye_Left].colorTex, depthFramebuffer2[vr::Eye_Right].colorTex,
		states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
	editorWindow.getVRView()->setInputFBOs(submitFramebuffer2[vr::Eye_Left].FBO,
		submitFramebuffer2[vr::Eye_Right].FBO, states->HMDRenderSizePerEye);
	std::array<vr::Texture_t, 2> submitVRTex2 = {
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Left].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma},
		vr::Texture_t{(void*)(uintptr_t)submitFramebuffer2[vr::Eye_Right].colorTex, vr::TextureType_OpenGL, vr::ColorSpace_Gamma}
	};

	{
		states->camera.setHeadPos(
			glm::vec3(states->subrgn.halfW,
				states->subrgn.halfH,
				states->subrgn.halfD * 2.5f));

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
	}

	states->pathManager.addPath();

	Math::printGLMMat4(states->eyeToHMD2[vr::Eye_Left], "HMDToEye L");
	Math::printGLMMat4(states->eyeToHMD2[vr::Eye_Right], "HMDToEye R");
	Math::printGLMMat4(states->projection2[vr::Eye_Left], "Projection L");
	Math::printGLMMat4(states->projection2[vr::Eye_Right], "Projection R");
	printf("Render %d x %d pixels per Eye\n", states->HMDRenderSizePerEye[0],
		states->HMDRenderSizePerEye[1]);

	std::array<glm::mat4, 2> VP2;
	std::array<glm::mat4, 2> gizmoMVP2;
	std::array<glm::mat4, 2> annotationMVP2;
	std::array<std::array<glm::mat4, 2>, 2> handMVP22;
	while (states->canRun)
	{
		qtApp.processEvents();
		editorWindow.getVRView()->makeCurrent();

		qtEvntHndler->update();
		if (states->canVRRun)
			vrEvntHndler->update();
		
		static auto identity = glm::identity<glm::mat4>();
		{
			auto& handZ = states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].transform[2];
			auto& handPos = states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].transform[3];
			glm::vec3 ballPos = handPos - ANNO_BALL_DIST_FROM_HAND * handZ;
			intrctParam.dat.ball.startPos.x = ballPos.x - ANNO_BALL_RADIUS;
			intrctParam.dat.ball.startPos.y = ballPos.y - ANNO_BALL_RADIUS;
			intrctParam.dat.ball.startPos.z = ballPos.z - ANNO_BALL_RADIUS;
			annotationBall.transform = glm::translate(glm::identity<glm::mat4>(), ballPos);
			annotationBall.transform = glm::scale(annotationBall.transform, glm::vec3{ ANNO_BALL_DIAMETER });
		}
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			VP2[eyeIdx] = states->projection2[eyeIdx]
				* states->camera.getViewMat(eyeIdx);
			gizmoMVP2[eyeIdx] = VP2[eyeIdx] * gizmo.transform;
			annotationMVP2[eyeIdx] = VP2[eyeIdx] * annotationBall.transform;
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

			shaders.depthShader.program.bind();
			glUniformMatrix4fv(
				shaders.depthShader.matPos, 1, GL_FALSE, (GLfloat*)&gizmoMVP2[eyeIdx]);
			gizmo.model->draw();

			if (states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].show)
			{
				glUniformMatrix4fv(
					shaders.depthShader.matPos, 1, GL_FALSE, (GLfloat*)&annotationMVP2[eyeIdx]);
				annotationBall.model->draw();
			}

			VRContext::forHandsDo([&](uint8_t hndIdx) {
				if (!states->hand2[hndIdx].show) return;
				glUniformMatrix4fv(
					shaders.depthShader.matPos, 1, GL_FALSE,
					(GLfloat*)&handMVP22[eyeIdx][hndIdx]);
				states->hand2[hndIdx].model->draw();
				});

			shaders.zeroDepthShader.program.bind();
			glUniformMatrix4fv(
				shaders.colorShader.matPos, 1, GL_FALSE, (GLfloat*)&VP2[eyeIdx]);
			glPointSize(5.f);
			glBindVertexArray(intrctPoint.VAO);
			glDrawArrays(GL_POINTS, 0, 1);
			glBindVertexArray(0);

			states->pathManager.draw(
				shaders.pathDepthShader.program.programId(),
				shaders.pathDepthShader.matPos, VP2[eyeIdx]);
			});

		glEnable(GL_MULTISAMPLE);
		glClearColor(0.f, 0.f, 0.f, 1.f); // restore
		VRContext::forEyesDo([&](uint8_t eyeIdx) {
			glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer2[eyeIdx].FBO);
			glViewport(0, 0, states->HMDRenderSizePerEye[0], states->HMDRenderSizePerEye[1]);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			shaders.colorShader.program.bind();
			glUniformMatrix4fv(
				shaders.colorShader.matPos, 1, GL_FALSE, (GLfloat*)&gizmoMVP2[eyeIdx]);
			gizmo.model->draw();

			if (states->hand2[static_cast<uint8_t>(VRContext::HandEnum::Right)].show)
			{
				glUniformMatrix4fv(
					shaders.colorShader.matPos, 1, GL_FALSE, (GLfloat*)&annotationMVP2[eyeIdx]);
				annotationBall.model->draw();
			}

			glUniformMatrix4fv(
				shaders.colorShader.matPos, 1, GL_FALSE, (GLfloat*)&VP2[eyeIdx]);
			glPointSize(5.f);
			glBindVertexArray(intrctPoint.VAO);
			glDrawArrays(GL_POINTS, 0, 1);
			glBindVertexArray(0);

			VRContext::forHandsDo([&](uint8_t hndIdx) {
				if (!states->hand2[hndIdx].show) return;
				shaders.diffuseShader.program.bind();
				glUniformMatrix4fv(
					shaders.diffuseShader.matPos, 1, GL_FALSE,
					(GLfloat*)&handMVP22[eyeIdx][hndIdx]);
				states->hand2[hndIdx].model->draw();
				});

			states->pathManager.draw(
				shaders.pathColorShader.program.programId(),
				shaders.pathColorShader.matPos, VP2[eyeIdx],
				shaders.pathColorShader.colorPos);
			});
		
		{
			auto [right, forward, up, lftEyePos, rhtEyePos] =
				states->camera.getRFUP2();
			glm::mat4 rotation(
				right.x, right.y, right.z, 0,
				up.x, up.y, up.z, 0,
				-forward.x, -forward.y, -forward.z, 0,
				0, 0, 0, 1.f);
			states->renderer->setCamera(
				{ lftEyePos, rhtEyePos, rotation,
				states->unProjection2[vr::Eye_Left],states->unProjection2[vr::Eye_Right],
				states->nearClip, states->farClip });
		}
		if (states->subrgnChanged)
		{
			states->renderer->setSubregion(states->subrgn);
			states->subrgnChanged = false;
		}
		states->renderer->setInteractionParam(intrctParam);
		states->renderer->render(&states->intrctPos, states->renderTar);
		glBindBuffer(GL_ARRAY_BUFFER, intrctPoint.VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat) * 3, &states->intrctPos);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		shaders.diffuseShader.program.bind();
		glUniformMatrix4fv(
			shaders.diffuseShader.matPos, 1, GL_FALSE, (GLfloat*)&identity);
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
