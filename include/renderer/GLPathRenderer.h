#ifndef KOUEK_GL_PATH_RENDERER_H
#define KOUEK_GL_PATH_RENDERER_H

#include <limits>

#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>

namespace kouek
{
	class GLPathRenderer
	{
	public:
		inline static constexpr GLuint MAX_VERT_NUM = 10000;
		inline static constexpr GLuint MAX_PATH_NUM = 100;
		inline static float rootVertSize = 5.f;
		inline static float endVertSize = 3.f;
		inline static float lineWidth = 1.f;
		inline static float selectVertSize = 5.f;
		inline static float selectedVertSize = 5.f;
		inline static glm::vec3 selectedVertColor{ 1.f, .5f, 1.f };

	private:
		inline static constexpr GLubyte VERT_DAT_POS_NUM = 3;
		inline static constexpr GLubyte VERT_DAT_ID_NUM = 4;
		inline static constexpr GLsizeiptr VERT_DAT_POS_SIZE
			= sizeof(GLfloat) * VERT_DAT_POS_NUM;
		inline static constexpr GLsizeiptr VERT_DAT_ID_SIZE
			= sizeof(GLfloat) * VERT_DAT_ID_NUM;
		inline static constexpr GLsizeiptr VERT_DAT_STRIDE
			= VERT_DAT_POS_SIZE + VERT_DAT_ID_SIZE;

#define VERTEX_ARRAY_DEF \
		glVertexAttribPointer(0, VERT_DAT_POS_NUM, GL_FLOAT, GL_FALSE,\
			VERT_DAT_STRIDE, nullptr);\
		glEnableVertexAttribArray(0);\
		glVertexAttribPointer(1, VERT_DAT_ID_NUM, GL_FLOAT, GL_FALSE,\
			VERT_DAT_STRIDE, (const void*)VERT_DAT_POS_SIZE);\
		glEnableVertexAttribArray(1);\

		struct SubPath
		{
			bool needUpload = true;
			GLuint VAO = 0, EBO = 0;
			GLuint vertGPUCap = 5;
			std::vector<GLuint> verts;
			SubPath(GLuint startVertID)
			{
				verts.emplace_back(startVertID);

				glGenVertexArrays(1, &VAO);
				glBindVertexArray(VAO);
				VERTEX_ARRAY_DEF;
				glGenBuffers(1, &EBO);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * vertGPUCap,
					nullptr, GL_DYNAMIC_DRAW);
				glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, 1, &startVertID);
				glBindVertexArray(0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}
			~SubPath()
			{
				glDeleteVertexArrays(1, &VAO);
				glDeleteBuffers(1, &EBO);
			}
			inline const auto getVertexIDs() const
			{
				return verts;
			}
			inline void addVertex(GLuint vertID)
			{
				verts.emplace_back(vertID);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				if (vertGPUCap < verts.size())
				{
					vertGPUCap = vertGPUCap << 1;
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * vertGPUCap,
						nullptr, GL_DYNAMIC_DRAW);
					glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
						sizeof(GLuint) * verts.size(), verts.data());
				}
				else
					glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,
						sizeof(GLuint) * (verts.size() - 1),
						sizeof(GLuint), &vertID);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}
			inline void upload()
			{
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				if (vertGPUCap < verts.size())
				{
					vertGPUCap = vertGPUCap << 1;
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * vertGPUCap,
						nullptr, GL_DYNAMIC_DRAW);
				}
				glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
					sizeof(GLuint) * verts.size(), verts.data());
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
				needUpload = false;
			}
			inline void drawLineStrips()
			{
				glBindVertexArray(VAO);
				glDrawElements(GL_LINE_STRIP, verts.size(),
					GL_UNSIGNED_INT, 0);
			}
			inline void drawVerts()
			{
				glBindVertexArray(VAO);
				glDrawElements(GL_POINTS, verts.size(),
					GL_UNSIGNED_INT, 0);
			}
			inline void drawEndVerts()
			{
				glBindVertexArray(VAO);
				glDrawElements(GL_POINTS, 1, GL_UNSIGNED_INT, 0);
				glDrawElements(GL_POINTS, 1, GL_UNSIGNED_INT,
					(const void*)(sizeof(GLuint) * (verts.size() - 1)));
			}
		};
		struct Path
		{
			GLuint rootID;
			GLuint VAO;
			glm::vec3 color;
			std::unordered_map<GLuint, SubPath> subPaths;
			std::queue<GLuint> recycledSubpathIDs;
			Path(
				const glm::vec3& color,
				GLuint rootID)
				:color(color), rootID(rootID)
			{
				glGenVertexArrays(1, &VAO);
				glBindVertexArray(VAO);
				VERTEX_ARRAY_DEF;
				glBindVertexArray(0);
			}
			~Path()
			{
				glDeleteVertexArrays(1, &VAO);
			}
			inline GLuint getRootID() const
			{
				return rootID;
			}
			inline const auto getSubPaths() const
			{
				return subPaths;
			}
			inline GLuint addSubPath(GLuint startVertID)
			{
				GLuint subPathID = recycledSubpathIDs.empty()
					? subPaths.size() : [&]() -> GLuint {
					GLuint ret = recycledSubpathIDs.front();
					recycledSubpathIDs.pop();
					return ret;
				}();
				subPaths.emplace(std::piecewise_construct,
					std::forward_as_tuple(subPathID),
					std::forward_as_tuple(startVertID));
				return subPathID;
			}
			inline void drawRoot()
			{
				glBindVertexArray(VAO);
				glDrawArrays(GL_POINTS, rootID, 1);
			}
		};
		GLuint VBO = 0, slctVertVAO;
		GLuint selectedPathID = std::numeric_limits<GLuint>::max();
		GLuint selectedVertID = std::numeric_limits<GLuint>::max();
		GLuint selectedSubPathID = std::numeric_limits<GLuint>::max();
		std::vector<glm::vec3> verts;
		std::vector<GLuint> pathIDOfVerts;
		std::unordered_map<GLuint, Path> paths;
		std::queue<GLuint> availableVertIDs;
		std::queue<GLuint> availablePathIDs;

	public:
		GLPathRenderer()
		{
			glGenVertexArrays(1, &slctVertVAO);
			glBindVertexArray(slctVertVAO);
			glGenBuffers(1, &VBO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferStorage(GL_ARRAY_BUFFER, VERT_DAT_STRIDE * MAX_VERT_NUM,
				nullptr, GL_DYNAMIC_STORAGE_BIT);
			VERTEX_ARRAY_DEF;
			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			for (GLuint id = 0; id < MAX_PATH_NUM; ++id)
				availablePathIDs.emplace(id);
			for (GLuint id = 0; id < MAX_VERT_NUM; ++id)
				availableVertIDs.emplace(id);
			verts.resize(MAX_VERT_NUM);
			pathIDOfVerts.resize(MAX_VERT_NUM);
		}

#undef VERTEX_ARRAY_DEF

		~GLPathRenderer()
		{
			glDeleteBuffers(1, &VBO);
			glDeleteVertexArrays(1, &slctVertVAO);
		}
		GLPathRenderer(const GLPathRenderer&) = delete;
		GLPathRenderer(GLPathRenderer&&) = delete;
		GLPathRenderer& operator=(const GLPathRenderer&) = delete;
		GLPathRenderer& operator=(GLPathRenderer&&) = delete;
		inline GLuint getSelectedPathID() const
		{
			return selectedPathID;
		}
		inline GLuint getSelectedSubPathID() const
		{
			return selectedSubPathID;
		}
		inline GLuint getSelectedVertID() const
		{
			return selectedVertID;
		}
		inline GLuint getPathIDOf(GLuint vertID) const
		{
			return pathIDOfVerts[vertID];
		}
		inline const auto getPaths() const
		{
			return paths;
		}
		inline const auto getVertexPositions() const
		{
			return verts;
		}
		inline GLuint addPath(
			const glm::vec3& color,
			const glm::vec3& rootPos)
		{
			GLuint pathID = availablePathIDs.front();
			availablePathIDs.pop();
			GLuint rootID = availableVertIDs.front();
			availableVertIDs.pop();

			pathIDOfVerts[rootID] = pathID;

			verts[rootID] = rootPos;
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			GLintptr offs = VERT_DAT_STRIDE * rootID;
			glBufferSubData(GL_ARRAY_BUFFER,
				offs, VERT_DAT_POS_SIZE, &rootPos);
			std::array<GLfloat, 4> id4{
				(float)((rootID & 0x000000ff) >> 0) / 255.f,
				(float)((rootID & 0x0000ff00) >> 8) / 255.f,
				(float)((rootID & 0x00ff0000) >> 16) / 255.f,
				(float)((rootID & 0xff000000) >> 24) / 255.f
			};
			glBufferSubData(GL_ARRAY_BUFFER,
				offs + VERT_DAT_POS_SIZE,
				VERT_DAT_ID_SIZE, id4.data());
			paths.emplace(std::piecewise_construct,
				std::forward_as_tuple(pathID),
				std::forward_as_tuple(color, rootID));
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			return pathID;
		}
		inline void startPath(GLuint pathID)
		{
			selectedPathID = pathID;
		}
		inline void endPath()
		{
			selectedPathID = selectedVertID = selectedSubPathID
				= std::numeric_limits<GLuint>::max();
		}
		inline void deletePath()
		{
			paths.erase(selectedPathID);
			selectedPathID = selectedVertID = selectedSubPathID
				= std::numeric_limits<GLuint>::max();
		}
		inline GLuint addSubPath()
		{
			Path& path = paths.at(selectedPathID);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			GLuint subPathID = path.addSubPath(
				selectedVertID == std::numeric_limits<GLuint>::max()
				? path.rootID : selectedVertID);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			return subPathID;
		}
		inline void startSubPath(GLuint subPathID)
		{
			selectedSubPathID = subPathID;
		}
		inline void endSubPath()
		{
			selectedVertID = selectedSubPathID
				= std::numeric_limits<GLuint>::max();
		}
		inline GLuint addVertex(const glm::vec3& pos)
		{
			GLuint vertID = availableVertIDs.front();
			availableVertIDs.pop();
			
			pathIDOfVerts[vertID] = selectedPathID;

			verts[vertID] = pos;
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			GLintptr offs = VERT_DAT_STRIDE * vertID;
			glBufferSubData(GL_ARRAY_BUFFER,
				offs, VERT_DAT_POS_SIZE, &pos);
			std::array<GLfloat, 4> id4{
				(float)((vertID & 0x000000ff) >> 0) / 255.f,
				(float)((vertID & 0x0000ff00) >> 8) / 255.f,
				(float)((vertID & 0x00ff0000) >> 16) / 255.f,
				(float)((vertID & 0xff000000) >> 24) / 255.f
			};
			glBufferSubData(GL_ARRAY_BUFFER,
				offs + VERT_DAT_POS_SIZE,
				VERT_DAT_ID_SIZE, id4.data());
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			auto& subPath = paths.at(selectedPathID)
				.subPaths.at(selectedSubPathID);
			subPath.addVertex(vertID);

			return vertID;
		}
		inline void startVertex(GLuint vertID)
		{
			selectedVertID = vertID;
		}
		inline void endVertex(GLuint vertID)
		{
			selectedVertID = std::numeric_limits<GLuint>::max();
		}
		inline void draw(GLint colUniformPos = -1)
		{
			// draw selected vert
			if (colUniformPos != -1
				&& selectedVertID != std::numeric_limits<GLuint>::max())
			{
				glPointSize(selectedVertSize);
				glUniform3fv(colUniformPos, 1, (const float*)&selectedVertColor);
				glBindVertexArray(slctVertVAO);
				glDrawArrays(GL_POINTS, selectedVertID, 1);
			}
			for (auto& [id, path] : paths)
			{
				if (colUniformPos != -1)
					glUniform3fv(colUniformPos, 1, (const float*)&path.color);
				// upload GPU needed data
				for (auto& [id, subPath] : path.subPaths)
					if (subPath.needUpload)
						subPath.upload();
				// draw lines
				glLineWidth(lineWidth);
				for (auto& [id, subPath] : path.subPaths)
					subPath.drawLineStrips();
				// draw end verts
				glPointSize(endVertSize);
				for (auto& [id, subPath] : path.subPaths)
					subPath.drawEndVerts();
				// draw root vert
				glPointSize(rootVertSize);
				path.drawRoot();
			}
			glLineWidth(1.f);
			glPointSize(1.f);
			glBindVertexArray(0);
		}
		inline void drawVertIDs()
		{
			for (auto& [id, path] : paths)
			{
				// upload GPU needed data
				for (auto& [id, subPath] : path.subPaths)
					if (subPath.needUpload)
						subPath.upload();
				// draw verts
				glPointSize(20.f);
				for (auto& [id, subPath] : path.subPaths)
					subPath.drawVerts();
			}
		}
	};
}

#endif // !KOUEK_GL_PATH_RENDERER_H
