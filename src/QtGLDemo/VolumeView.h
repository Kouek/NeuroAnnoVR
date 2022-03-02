#ifndef KOUEK_VOLUME_VIEW_H
#define KOUEK_VOLUME_VIEW_H

#include <vector>

// ensure glad.h is ahead of any QtOpenGL header
#include <renderer/Renderer.h>

#include <QtWidgets/qopenglwidget.h>
#include <QtGui/qevent.h>
#include <QtGui/qopenglshaderprogram.h>
#include <QtGui/qopenglbuffer.h>
#include <QtGui/qopenglvertexarrayobject.h>

#include <camera/FPSCamera.h>
#include <util/RenderObj.h>

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
	private:
		FPSCamera camera;
		glm::mat4 projection;

		GLint colorShaderMatrixPos, diffuseShaderMatrixPos;

		QOpenGLShaderProgram colorShader, diffuseShader;

		struct
		{
			glm::mat4 transform = Math::IDENTITY;
			std::unique_ptr<WireFrame> model;
		}gizmo;

		struct
		{
			GLuint PBO = 0;
			GLuint tex = 0;
			GLuint VBO = 0, EBO = 0, VAO = 0;
			CompVolumeRenderer::Subregion subrgn;
			std::unique_ptr<kouek::CompVolumeMonoEyeRenderer> renderer;
		}volumeRender;

		struct
		{
			QPoint lastCursorPos;
			Qt::MouseButton lastCursorBtn = Qt::NoButton;
		}state;

	public:
		VolumeView(QWidget* parent = Q_NULLPTR)
			: QOpenGLWidget(parent),
			camera(glm::vec3{ 5.5f,5.5f,5.5f }, glm::vec3{ 0,0,0 })
		{
			setFocusPolicy(Qt::StrongFocus);

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
		}
		~VolumeView()
		{
			volumeRender.renderer->unregisterOutputGLPBO();
		}

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
				glm::rotate(Math::IDENTITY, rad, glm::vec3{ 0,1.f,0 });
			onSubregionUpdated();
		}

	private:
		inline void onSubregionUpdated()
		{
			auto& subrgn = volumeRender.subrgn;
			volumeRender.renderer->setSubregion(subrgn);
			auto translate = glm::translate(Math::IDENTITY,
				glm::vec3{ -subrgn.halfW,-subrgn.halfH,-subrgn.halfD });
			auto invTranslate = glm::translate(Math::IDENTITY,
				glm::vec3{ subrgn.halfW,subrgn.halfH,subrgn.halfD });
			gizmo.transform = glm::scale(Math::IDENTITY,
				glm::vec3{ subrgn.halfW * 2,subrgn.halfH * 2,subrgn.halfD * 2 });
			gizmo.transform = translate * gizmo.transform;
			gizmo.transform = subrgn.rotation * gizmo.transform;
			gizmo.transform = invTranslate * gizmo.transform;
			update();
		}

	protected:
		void initializeGL() override
		{
			// load GL funcs
			int hasInit = gladLoadGL();
			assert(hasInit);

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
				gizmo.transform = glm::identity<glm::mat4>();
			}

			// volumeRender
			{
				glGenVertexArrays(1, &volumeRender.VAO);
				glBindVertexArray(volumeRender.VAO);
				glGenBuffers(1, &volumeRender.VBO);
				glBindBuffer(GL_ARRAY_BUFFER, volumeRender.VBO);
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

				glGenBuffers(1, &volumeRender.EBO);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, volumeRender.EBO);
				{
					GLushort idxes[] = { 0, 1, 3, 0, 3, 2 };
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * 6, idxes, GL_STATIC_DRAW);
				}
				glBindVertexArray(0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

				glGenBuffers(1, &volumeRender.PBO);
				glGenTextures(1, &volumeRender.tex);
			}

			// sync default val from this level to deeper logic
			glClearColor(0, 0, 0, 1.f);
			onSubregionUpdated();
			onCameraUpdated();
		}

		void resizeGL(int w, int h) override
		{
			projection = glm::perspectiveFov(
				glm::radians(90.f), (float)w, (float)h, .1f, 100.f);
			
			volumeRender.renderer->unregisterOutputGLPBO();
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, volumeRender.PBO);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(GLuint) * w * h, NULL,
				GL_STREAM_COPY);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			volumeRender.renderer->registerOutputGLPBO(volumeRender.PBO, w, h);

			glBindTexture(GL_TEXTURE_2D, volumeRender.tex);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
			glTexImage2D(
				GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
				GL_UNSIGNED_INT_8_8_8_8, NULL);
			glBindTexture(GL_TEXTURE_2D, 0);

			glViewport(0, 0, w, h);
		}

		void paintGL() override
		{
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			volumeRender.renderer->render();
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, volumeRender.PBO);
			{
				glBindTexture(GL_TEXTURE_2D, volumeRender.tex);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), GL_RGBA,
					GL_UNSIGNED_INT_8_8_8_8, (const void*)0);
			}
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			glDisable(GL_DEPTH_TEST);

			diffuseShader.bind();
			static auto identity = glm::identity<glm::mat4>();
			glUniformMatrix4fv(
				diffuseShaderMatrixPos, 1, GL_FALSE,
				(GLfloat*)&identity);
			{
				glBindVertexArray(volumeRender.VAO);
				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void*)0);
				glBindVertexArray(0);
				glBindTexture(GL_TEXTURE_2D, 0);
			}

			glEnable(GL_DEPTH_TEST);

			colorShader.bind();
			glUniformMatrix4fv(
				colorShaderMatrixPos, 1, GL_FALSE,
				(GLfloat*)&(projection * camera.getViewMat() * gizmo.transform));
			printGLMMat4(projection);
			printGLMMat4(camera.getViewMat());
			gizmo.model->draw();
		}

	private:
		inline void printGLMMat4(const glm::mat4& mat4)
		{
			printf("[\n%f\t%f\t%f\t%f\n", mat4[0][0], mat4[1][0], mat4[2][0], mat4[3][0]);
			printf("%f\t%f\t%f\t%f\n", mat4[0][1], mat4[1][1], mat4[2][1], mat4[3][1]);
			printf("%f\t%f\t%f\t%f\n", mat4[0][2], mat4[1][2], mat4[2][2], mat4[3][2]);
			printf("%f\t%f\t%f\t%f\n]\n", mat4[0][3], mat4[1][3], mat4[2][3], mat4[3][3]);
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
			constexpr std::array<float, 3> ROTATE_SENSITY = { .1f,.1f };
			auto& pos = event->pos();
			auto difPos = pos - state.lastCursorPos;
			switch (state.lastCursorBtn)
			{
			case Qt::LeftButton:
				camera.rotate(
					difPos.x() * -ROTATE_SENSITY[0],
					difPos.y() * -ROTATE_SENSITY[1]);
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
			constexpr float MOVE_SENSITY = .1f;
			switch (event->key())
			{
			case Qt::Key_W:
				camera.move(0, 0, MOVE_SENSITY);
				onCameraUpdated();
				break;
			case Qt::Key_A:
				camera.move(-MOVE_SENSITY, 0, 0);
				onCameraUpdated();
				break;
			case Qt::Key_S:
				camera.move(0, 0, -MOVE_SENSITY);
				onCameraUpdated();
				break;
			case Qt::Key_D:
				camera.move(MOVE_SENSITY, 0, 0);
				onCameraUpdated();
				break;
			default:
				break;
			}
		}

	private:
		void onCameraUpdated()
		{
			auto [right, forward, up, pos] = camera.getRFUP();
			glm::mat4 rotation(
				right.x, right.y, right.z, 0,
				up.x, up.y, up.z, 0,
				-forward.x, -forward.y, -forward.z, 0,
				0, 0, 0, 1.f);
			volumeRender.renderer->setCamera(
				pos, rotation, Math::inverseProjective(projection));
			update();
		}
	};
}

#endif // !KOUEK_VOLUME_VIEW_H
