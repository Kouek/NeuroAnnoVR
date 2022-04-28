#ifndef KOUEK_GL_VIEW_H
#define KOUEK_GL_VIEW_H

#include <util/SWCConverter.h>

#include <QtGUI/qopenglshaderprogram.h>
#include <QtGui/qevent.h>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qopenglwidget.h>

namespace kouek
{
	class GLView : public QOpenGLWidget
	{
		Q_OBJECT

	private:
		glm::mat4 projection;
		glm::mat4 camera;

		std::unique_ptr<FileSWC> swc;
		std::unique_ptr<GLPathRenderer> pathRenderer;

		struct
		{
			QOpenGLShaderProgram program;
			GLint matPos, colorPos;
		}pathColorShader;

	public:
		GLView(QWidget* parent = Q_NULLPTR)
			: QOpenGLWidget(parent)
		{
			QSurfaceFormat surfaceFmt;
			surfaceFmt.setDepthBufferSize(24);
			surfaceFmt.setStencilBufferSize(8);
			surfaceFmt.setVersion(4, 5);
			surfaceFmt.setProfile(QSurfaceFormat::CoreProfile);
			setFormat(surfaceFmt);
		}

	protected:
		void initializeGL() override
		{
			// load GL funcs
			int hasInit = gladLoadGL();
			assert(hasInit);

			camera = glm::lookAt(glm::vec3{ 0,0,1.f },
				glm::vec3{ 0 }, glm::vec3{ 0,1.f,0 });

			{
				const char* vertShaderCode =
					"#version 410 core\n"
					"uniform mat4 matrix;\n"
					"layout(location = 0) in vec3 position;\n"
					"layout(location = 1) in vec4 id;\n"
					"void main()\n"
					"{\n"
					"	gl_Position = matrix * vec4(position.xyz, 1.0);\n"
					"}\n";
				const char* fragShaderCode =
					"#version 410 core\n"
					"uniform vec3 color;\n"
					"out vec4 outputColor;\n"
					"void main()\n"
					"{\n"
					"    outputColor = vec4(color, 0.5);\n"
					"}\n";
				pathColorShader.program.addShaderFromSourceCode(
					QOpenGLShader::Vertex, vertShaderCode);
				pathColorShader.program.addShaderFromSourceCode(
					QOpenGLShader::Fragment, fragShaderCode);
				pathColorShader.program.link();
				assert(pathColorShader.program.isLinked());

				pathColorShader.matPos = pathColorShader.program.uniformLocation("matrix");
				assert(pathColorShader.matPos != -1);
				pathColorShader.colorPos = pathColorShader.program.uniformLocation("color");
				assert(pathColorShader.colorPos != -1);
			}

			pathRenderer = std::make_unique<GLPathRenderer>();
			// 0
			// +-1-2
			// +-3-4-5
			// | +-6-7
			// +-8
			//   +-9-10
			auto id = pathRenderer->addPath(glm::vec3{ 1.f,0,1.f },
				glm::vec3{ 0,.1f,-1.f });
			pathRenderer->startPath(id);
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id = pathRenderer->addVertex(glm::vec3{ .5f,0,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ .75f,0,-1.f });
			pathRenderer->endSubPath();
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			auto id2 = id = pathRenderer->addVertex(glm::vec3{ .5f,.5f,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ .75f,.5f,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ .825f,.5f,-1.f });
			pathRenderer->endSubPath();
			pathRenderer->startVertex(id2);
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id = pathRenderer->addVertex(glm::vec3{ .75f,.75f,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ .75f,.75f,-1.f });
			pathRenderer->endSubPath();
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id2 = id = pathRenderer->addVertex(glm::vec3{ .5f,.75f,-1.f });
			pathRenderer->endSubPath();
			pathRenderer->startVertex(id2);
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id = pathRenderer->addVertex(glm::vec3{ .75f,.825f,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ .825f,.825f,-1.f });
			pathRenderer->endSubPath();
			pathRenderer->endPath();
			// 9
			// +-10-11
			// +-12-13-14
			// | +-15-16
			// +-17
			/*pathRenderer->addPath(glm::vec3{ 1.f,1.f,0 }, glm::vec3{ 0 });*/
			id = pathRenderer->addPath(glm::vec3{ 1.f,1.f,0 },
				glm::vec3{ 0,-.1f,-1.f });
			pathRenderer->startPath(id);
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id = pathRenderer->addVertex(glm::vec3{ -.5f,0,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ -.75f,0,-1.f });
			pathRenderer->endSubPath();
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id2 = id = pathRenderer->addVertex(glm::vec3{ -.5f,-.5f,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ -.75f,-.5f,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ -.825f,-.5f,-1.f });
			pathRenderer->endSubPath();
			pathRenderer->startVertex(id2);
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id = pathRenderer->addVertex(glm::vec3{ -.75f,-.75f,-1.f });
			pathRenderer->startVertex(id);
			id = pathRenderer->addVertex(glm::vec3{ -.75f,-.75f,-1.f });
			pathRenderer->endSubPath();
			id = pathRenderer->addSubPath();
			pathRenderer->startSubPath(id);
			id = pathRenderer->addVertex(glm::vec3{ -.5f,-.75f,-1.f });
			pathRenderer->endSubPath();
			pathRenderer->endPath();
		}

		void resizeGL(int w, int h) override
		{
			projection = glm::perspectiveFov(
				glm::radians(60.f), (float)w, (float)h, .1f, 10.f);
		}

		void paintGL() override
		{
			glm::mat4 VP = projection * camera;
			glClearColor(0, 0, 0, 1.f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			pathColorShader.program.bind();
			glUniformMatrix4fv(
				pathColorShader.matPos, 1, GL_FALSE, (GLfloat*)&VP);
			pathRenderer->draw(pathColorShader.colorPos);
		}

		void keyPressEvent(QKeyEvent* e) override
		{
			switch (e->key())
			{
			case Qt::Key_W:
				swc = std::make_unique<FileSWC>("./SWC.txt");
				SWCConverter::fromGLPathRendererToSWC(
					*pathRenderer, *swc);
				swc.reset();
				break;
			case Qt::Key_C:
				pathRenderer->clear();
				break;
			case Qt::Key_R:
				pathRenderer->clear();
				swc = std::make_unique<FileSWC>("./SWC.txt");
				SWCConverter::appendSWCToGLPathRenderer(
					*swc, *pathRenderer);
				swc.reset();
				break;
			}
			update();
		}
	};
}

#endif // !KOUEK_GL_VIEW_H
