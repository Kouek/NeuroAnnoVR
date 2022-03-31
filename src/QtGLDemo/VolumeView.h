#ifndef KOUEK_VOLUME_VIEW_H
#define KOUEK_VOLUME_VIEW_H

#include <vector>

// ensure glad.h is ahead of any QtOpenGL header
#include <renderer/Renderer.h>

#include <QtWidgets/qopenglwidget.h>
#include <QtGui/qevent.h>
#include <QtGui/qopenglshaderprogram.h>

#include <CMakeIn.h>

#include <camera/FPSCamera.h>
#include <util/RenderObj.h>
#include <util/VolumeCfg.h>

#define GL_CHECK \
         {       \
            GLenum gl_err; \
            if((gl_err=glGetError())!=GL_NO_ERROR){     \
            spdlog::error("OpenGL error: {0} caused before  on line {1} of file:{2}",static_cast<unsigned int>(gl_err),__LINE__,__FILE__);     \
            }\
         }

namespace kouek
{
	class VolumeView : public QOpenGLWidget
	{
		Q_OBJECT

	private:
		GLint depthShaderMatrixPos, colorShaderMatrixPos, diffuseShaderMatrixPos;

		float rotateSensity = .1f, moveSensity = .01f;
		float nearClip = .01f, farClip = 10.f;
		FPSCamera camera;
		glm::mat4 projection, unProjection;

		QOpenGLShaderProgram depthShader, colorShader, diffuseShader;

		struct
		{
			GLuint FBO, colorTex, depthRBO;
		}depthFramebuffer{ 0 };

		struct
		{
			GLuint FBO, colorTex, depthRBO;
		}colorFramebuffer{ 0 };

		struct
		{
			glm::mat4 transform = glm::identity<glm::mat4>();
			std::unique_ptr<WireFrame> model;
		}gizmo;

		struct
		{
			GLuint VAO, VBO, EBO;
		}screenQuad;

		struct
		{
			GLuint tex;
			CompVolumeRenderer::Subregion subrgn;
			std::shared_ptr<vs::CompVolume> volume;
			std::unique_ptr<kouek::CompVolumeMonoEyeRenderer> renderer;
		}volumeRender;

		struct
		{
			bool CTRLPressed = false;
			QPoint lastCursorPos;
			Qt::MouseButton lastCursorBtn = Qt::NoButton;
		}state;

	public:
		VolumeView(QWidget* parent = Q_NULLPTR)
			: QOpenGLWidget(parent),
			camera(glm::vec3{ .16f,.16f,.16f }, glm::zero<glm::vec3>())
		{
			setFocusPolicy(Qt::StrongFocus);
			setCursor(Qt::CrossCursor);

			{
				QSurfaceFormat surfaceFmt;
				surfaceFmt.setDepthBufferSize(24);
				surfaceFmt.setStencilBufferSize(8);
				surfaceFmt.setVersion(4, 5);
				surfaceFmt.setProfile(QSurfaceFormat::CoreProfile);
				setFormat(surfaceFmt);
			}

			{
				CompVolumeMonoEyeRenderer::CUDAParameter param;
				SetCUDACtx(0);
				param.ctx = GetCUDACtx();
				param.texUnitNum = 1;
				param.texUnitDim = { 1024,1024,1024 };
				volumeRender.renderer = CompVolumeMonoEyeRenderer::create(param);
			}

			try
			{
				std::string cfgPath(kouek::PROJECT_SOURCE_DIR);
				cfgPath.append("/cfg/VolumeCfg.json");
				kouek::VolumeConfig cfg(cfgPath.c_str());
				volumeRender.volume =
					vs::CompVolume::Load(cfg.getResourcePath().c_str());
				volumeRender.volume->SetSpaceX(cfg.getSpaceX());
				volumeRender.volume->SetSpaceY(cfg.getSpaceY());
				volumeRender.volume->SetSpaceZ(cfg.getSpaceZ());
				volumeRender.renderer->setVolume(volumeRender.volume);

				volumeRender.renderer->setStep(
					3000, cfg.getBaseSpace() * 0.3);
			}
			catch (std::exception& e)
			{
				spdlog::error("File: {0}, Line: {1}, Error: {2}", __FILE__,
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
				CompVolumeMonoEyeRenderer::LightParamter param;
				param.ka = 0.5f;
				param.kd = 0.8f;
				param.ks = 0.5f;
				param.shininess = 64.f;
				param.bkgrndColor = { .2f,.3f,.4f,.1f };
				volumeRender.renderer->setLightParam(param);
			}

			volumeRender.subrgn.center = { 3.24f,3.48f,5.21f };
			// sync default val from this level to deeper logic
			onSubregionUpdated();
			onCameraUpdated();
		}
		~VolumeView()
		{
			volumeRender.renderer->unregisterGLResource();
		}

	signals:
		void subregionMoved(const std::array<uint32_t, 3>& blockOfSubrgnCenter);

	public:
		inline void setSubregionHalfW(float hfW)
		{
			volumeRender.subrgn.halfW = hfW;
			onSubregionUpdated();
		}
		inline void setSubregionHalfH(float hfH)
		{
			volumeRender.subrgn.halfH = hfH;
			onSubregionUpdated();
		}
		inline void setSubregionHalfD(float hfD)
		{
			volumeRender.subrgn.halfD = hfD;
			onSubregionUpdated();
		}
		inline void setSubregionRotationY(float deg)
		{
			float rad = glm::radians(deg);
			volumeRender.subrgn.rotation =
				glm::rotate(glm::identity<glm::mat4>(), rad, glm::vec3{ 0,1.f,0 });
			onSubregionUpdated();
		}

	private:
		void onSubregionUpdated()
		{
			auto& subrgn = volumeRender.subrgn;

			// sync Subregion
			gizmo.transform = glm::scale(glm::identity<glm::mat4>(),
				glm::vec3{ subrgn.halfW * 2,subrgn.halfH * 2,subrgn.halfD * 2 });
			auto invTranslation = glm::translate(glm::identity<glm::mat4>(),
				glm::vec3{ -subrgn.halfW,-subrgn.halfH,-subrgn.halfD });
			auto translation = glm::translate(glm::identity<glm::mat4>(),
				glm::vec3{ subrgn.halfW,subrgn.halfH,subrgn.halfD });
			gizmo.transform = translation * subrgn.rotation
				* invTranslation * gizmo.transform;

			subrgn.fromWorldToSubrgn = Math::inversePose(
				translation * subrgn.rotation * invTranslation);
			volumeRender.renderer->setSubregion(subrgn);

			// sync with other logic
			auto [noPaddingBlockLen, padding, minLOD, maxLOD] =
				volumeRender.volume->GetBlockLength();
			noPaddingBlockLen -= 2 * padding;
			std::array subrgnInBlock{
				(uint32_t)(subrgn.center.x / volumeRender.volume->GetVolumeSpaceX() / noPaddingBlockLen),
				(uint32_t)(subrgn.center.y / volumeRender.volume->GetVolumeSpaceY() / noPaddingBlockLen),
				(uint32_t)(subrgn.center.z / volumeRender.volume->GetVolumeSpaceZ() / noPaddingBlockLen)
			};
			emit subregionMoved(subrgnInBlock);

			update();
		}

	protected:
		void initializeGL() override
		{
			// load GL funcs
			int hasInit = gladLoadGL();
			assert(hasInit);

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

			// colorShader
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

			// diffuseShader
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

			// gizmo
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
				gizmo.model = std::make_unique<WireFrame>(verts);
			}

			// depthFramebuffer and colorFramebuffer
			{
				glGenFramebuffers(1, &depthFramebuffer.FBO);
				glGenTextures(1, &depthFramebuffer.colorTex);
				glGenRenderbuffers(1, &depthFramebuffer.depthRBO);

				glGenFramebuffers(1, &colorFramebuffer.FBO);
				glGenTextures(1, &colorFramebuffer.colorTex);
				glGenRenderbuffers(1, &colorFramebuffer.depthRBO);
			}

			// screenQuad
			{
				glGenVertexArrays(1, &screenQuad.VAO);
				glBindVertexArray(screenQuad.VAO);
				glGenBuffers(1, &screenQuad.VBO);
				glBindBuffer(GL_ARRAY_BUFFER, screenQuad.VBO);
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

				glGenBuffers(1, &screenQuad.EBO);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, screenQuad.EBO);
				{
					GLushort idxes[] = { 0, 1, 3, 0, 3, 2 };
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * 6, idxes, GL_STATIC_DRAW);
				}
				glBindVertexArray(0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}

			// volumeRender
			{
				glGenTextures(1, &volumeRender.tex);
			}
		}

		void resizeGL(int w, int h) override
		{
			projection = glm::perspectiveFov(
				glm::radians(60.f), (float)w, (float)h, nearClip, farClip);
			unProjection = Math::inverseProjective(projection);
			onCameraUpdated();

			volumeRender.renderer->unregisterGLResource();
			{
				glBindTexture(GL_TEXTURE_2D, volumeRender.tex);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
					GL_RGBA, GL_UNSIGNED_BYTE, NULL);
				glBindTexture(GL_TEXTURE_2D, 0);
			}
			{
				glBindRenderbuffer(GL_RENDERBUFFER, depthFramebuffer.depthRBO);
				glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
				glBindRenderbuffer(GL_RENDERBUFFER, 0);

				glBindTexture(GL_TEXTURE_2D, depthFramebuffer.colorTex);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
					GL_RGBA, GL_FLOAT, (const void*)0);
				glBindTexture(GL_TEXTURE_2D, 0);

				glBindFramebuffer(GL_FRAMEBUFFER, depthFramebuffer.FBO);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
					GL_TEXTURE_2D, depthFramebuffer.colorTex, 0);
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
					GL_RENDERBUFFER, depthFramebuffer.depthRBO);
			}
			volumeRender.renderer->registerGLResource(
				volumeRender.tex, depthFramebuffer.colorTex, w, h);

			{
				glBindRenderbuffer(GL_RENDERBUFFER, colorFramebuffer.depthRBO);
				glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
				glBindRenderbuffer(GL_RENDERBUFFER, 0);

				glBindTexture(GL_TEXTURE_2D, colorFramebuffer.colorTex);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
					GL_RGBA, GL_UNSIGNED_BYTE, (const void*)0);
				glBindTexture(GL_TEXTURE_2D, 0);

				glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer.FBO);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
					GL_TEXTURE_2D, colorFramebuffer.colorTex, 0);
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
					GL_RENDERBUFFER, colorFramebuffer.depthRBO);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
		}

		void paintGL() override
		{
			glEnable(GL_DEPTH_TEST);
			glDisable(GL_BLEND);
			glBindFramebuffer(GL_FRAMEBUFFER, depthFramebuffer.FBO);
			glClearColor(1.f, 1.f, 1.f, 1.f); // area without frag corresp to FarClip
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			auto MVP = projection * camera.getViewMat() * gizmo.transform;
			{
				depthShader.bind();
				glUniformMatrix4fv(
					depthShaderMatrixPos, 1, GL_FALSE, (GLfloat*)&MVP);
				gizmo.model->draw();
			}

			glBindFramebuffer(GL_FRAMEBUFFER, colorFramebuffer.FBO);
			glClearColor(0.f, 0.f, 0.f, 1.f); // restore
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			{
				colorShader.bind();
				glUniformMatrix4fv(
					colorShaderMatrixPos, 1, GL_FALSE, (GLfloat*)&MVP);
				gizmo.model->draw();
			}

			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
			glClear(GL_COLOR_BUFFER_BIT);
			{
				volumeRender.renderer->render();

				diffuseShader.bind();
				static auto identity = glm::identity<glm::mat4>();
				glUniformMatrix4fv(
					diffuseShaderMatrixPos, 1, GL_FALSE,
					(GLfloat*)&identity);
				glBindVertexArray(screenQuad.VAO);
				{
					glBindTexture(GL_TEXTURE_2D, colorFramebuffer.colorTex);
					glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void*)0);
					glBindTexture(GL_TEXTURE_2D, volumeRender.tex);
					glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void*)0);
					glBindTexture(GL_TEXTURE_2D, 0);
				}
				glBindVertexArray(0);
			}
		}

	public:
		inline void setFPSCamera(
			const glm::vec3& eyePos,
			float moveSensity, float rotateSensity)
		{
			camera = kouek::FPSCamera(eyePos, glm::zero<glm::vec3>());
			this->moveSensity = moveSensity;
			this->rotateSensity = rotateSensity;
		}

	protected:
		void mousePressEvent(QMouseEvent* event) override
		{
			state.lastCursorBtn = event->button();
			switch (state.lastCursorBtn)
			{
			case Qt::LeftButton:
				state.lastCursorPos = event->pos();
				break;
			default:
				break;
			}
			event->accept();
		}

		void mouseMoveEvent(QMouseEvent* event) override
		{
			auto& pos = event->pos();
			auto difPos = pos - state.lastCursorPos;
			switch (state.lastCursorBtn)
			{
			case Qt::LeftButton:
				camera.rotate(
					difPos.x() * -rotateSensity,
					difPos.y() * -rotateSensity);
				state.lastCursorPos = pos;
				onCameraUpdated();
				break;
			default:
				break;
			}
			event->accept();
		}

		void keyPressEvent(QKeyEvent* event) override
		{
			switch (event->key())
			{
			case Qt::Key_Control:
				state.CTRLPressed = true;
				break;
			case Qt::Key_W:
				camera.move(0, 0, moveSensity);
				onCameraUpdated();
				break;
			case Qt::Key_A:
				camera.move(-moveSensity, 0, 0);
				onCameraUpdated();
				break;
			case Qt::Key_S:
				camera.move(0, 0, -moveSensity);
				onCameraUpdated();
				break;
			case Qt::Key_D:
				camera.move(moveSensity, 0, 0);
				onCameraUpdated();
				break;
			case Qt::Key_Left:
				volumeRender.subrgn.center += glm::vec3{ -moveSensity,0,0 };
				onSubregionUpdated();
				break;
			case Qt::Key_Right:
				volumeRender.subrgn.center += glm::vec3{ moveSensity,0,0 };
				onSubregionUpdated();
				break;
			case Qt::Key_Down:
				if (state.CTRLPressed)
					volumeRender.subrgn.center += glm::vec3{ 0,-moveSensity,0 };
				else
					volumeRender.subrgn.center += glm::vec3{ 0,0,-moveSensity };
				onSubregionUpdated();
				break;
			case Qt::Key_Up:
				if (state.CTRLPressed)
					volumeRender.subrgn.center += glm::vec3{ 0,moveSensity,0 };
				else
					volumeRender.subrgn.center += glm::vec3{ 0,0,moveSensity };
				onSubregionUpdated();
				break;
			default:
				break;
			}
		}

		void keyReleaseEvent(QKeyEvent* event)
		{
			switch (event->key())
			{
			case Qt::Key_Control:
				state.CTRLPressed = false;
				break;
			default:
				break;
			}
		}

	private:
		inline void onCameraUpdated()
		{
			auto [right, forward, up, pos] = camera.getRFUP();
			glm::mat4 rotation(
				right.x, right.y, right.z, 0,
				up.x, up.y, up.z, 0,
				-forward.x, -forward.y, -forward.z, 0,
				0, 0, 0, 1.f);
			
			volumeRender.renderer->setCamera(
				{ pos, rotation, unProjection, nearClip, farClip });
			update();
		}
	};
}

#endif // !KOUEK_VOLUME_VIEW_H
