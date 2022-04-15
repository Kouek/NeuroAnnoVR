#ifndef KOUEK_SHADERS
#define KOUEK_SHADERS

#include <glad/glad.h>
#include <QtGUI/qopenglshaderprogram.h>

namespace kouek
{
	struct Shaders
	{
		struct
		{
			GLint matPos;
			QOpenGLShaderProgram program;
		}depthShader;
		struct
		{
			GLint matPos;
			QOpenGLShaderProgram program;
		}zeroDepthShader;
		struct
		{
			GLint matPos;
			QOpenGLShaderProgram program;
		}pathDepthShader;
		struct
		{
			GLint matPos;
			QOpenGLShaderProgram program;
		}colorShader;
		struct
		{
			GLint matPos, colorPos;
			QOpenGLShaderProgram program;
		}pathColorShader;
		struct
		{
			GLint matPos;
			QOpenGLShaderProgram program;
		}diffuseShader;

		Shaders()
		{
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
				depthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Vertex, vertShaderCode);
				depthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Fragment, fragShaderCode);
				depthShader.program.link();
				assert(depthShader.program.isLinked());

				depthShader.matPos = depthShader.program.uniformLocation("matrix");
				assert(depthShader.matPos != -1);
			}
			// zeroDepthShader
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
					"    outputColor = vec4(0, 0, 0, 1.0);\n"
					"}\n";
				zeroDepthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Vertex, vertShaderCode);
				zeroDepthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Fragment, fragShaderCode);
				zeroDepthShader.program.link();
				assert(zeroDepthShader.program.isLinked());

				zeroDepthShader.matPos = depthShader.program.uniformLocation("matrix");
				assert(zeroDepthShader.matPos != -1);
			}
			// pathDepthShader
			{
				const char* vertShaderCode =
					"#version 410 core\n"
					"uniform mat4 matrix;\n"
					"layout(location = 0) in vec3 position;\n"
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
				pathDepthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Vertex, vertShaderCode);
				pathDepthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Fragment, fragShaderCode);
				pathDepthShader.program.link();
				assert(pathDepthShader.program.isLinked());

				pathDepthShader.matPos = pathDepthShader.program.uniformLocation("matrix");
				assert(pathDepthShader.matPos != -1);
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
				colorShader.program.addShaderFromSourceCode(
					QOpenGLShader::Vertex, vertShaderCode);
				colorShader.program.addShaderFromSourceCode(
					QOpenGLShader::Fragment, fragShaderCode);
				colorShader.program.link();
				assert(colorShader.program.isLinked());

				colorShader.matPos = colorShader.program.uniformLocation("matrix");
				assert(colorShader.matPos != -1);
			}
			// pathColorShader
			{
				const char* vertShaderCode =
					"#version 410 core\n"
					"uniform mat4 matrix;\n"
					"layout(location = 0) in vec3 position;\n"
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
					"	gl_Position = matrix * vec4(position.xyz, 1.0);\n"
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
				diffuseShader.program.addShaderFromSourceCode(
					QOpenGLShader::Vertex, vertShaderCode);
				diffuseShader.program.addShaderFromSourceCode(
					QOpenGLShader::Fragment, fragShaderCode);
				diffuseShader.program.link();
				assert(diffuseShader.program.isLinked());

				diffuseShader.matPos = diffuseShader.program.uniformLocation("matrix");
				assert(diffuseShader.matPos != -1);
			}
			/*{
				const char* vertShaderCode =
					"#version 410 core\n"
					"uniform mat4 matrix;\n"
					"uniform vec3 center;\n"
					"layout(location = 0) in vec3 position;\n"
					"layout(location = 1) in vec3 v3ColorIn;\n"
					"out int id;\n"
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
				depthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Vertex, vertShaderCode);
				depthShader.program.addShaderFromSourceCode(
					QOpenGLShader::Fragment, fragShaderCode);
				depthShader.program.link();
				assert(depthShader.program.isLinked());

				depthShader.matPos = depthShader.program.uniformLocation("matrix");
				assert(depthShader.matPos != -1);
			}*/
		}
	};
}

#endif // !KOUEK_SHADERS
