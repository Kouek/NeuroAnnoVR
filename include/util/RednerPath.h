#ifndef KOUEK_RENDER_PATH
#define KOUEK_RENDER_PATH

#include <stdexcept>

#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <glad/glad.h>

#include <glm/gtc/matrix_transform.hpp>

namespace kouek
{
    constexpr GLuint MAX_VERT_NUM = 1000;
    constexpr GLuint MAX_PATH_NUM = 100;
    constexpr float LINE_WIDTH = 4.f;
    constexpr float POINT_SIZE = 8.f;

    class RenderPath
    {
    private:
        int32_t id;
        GLuint vertVAO = 0, vertEBO = 0;
        GLuint segVAO = 0, segEBO = 0;
        bool needUpdate = false;

        std::array<float, 3> color;
        std::unordered_set<GLuint> verts;

        struct HashPreSucVBOIdx
        {
            size_t operator()(const std::array<GLuint, 2>& arr) const
            {
                static size_t hfBitLen = sizeof(GLuint) << 2;
                static size_t bitAndOp = std::numeric_limits<size_t>::max() << hfBitLen;
                return (arr[0] << hfBitLen) | (arr[1] & bitAndOp);
            }
        };
        struct EqualPreSucVBOIdx
        {
            bool operator()(
                const std::array<GLuint, 2>& a,
                const std::array<GLuint, 2>& b) const
            {
                if (a[0] == b[0] && a[1] == b[1]) return true;
                if (a[1] == b[0] && a[0] == b[1]) return true;
                return false;
            }
        };
        std::unordered_set<std::array<GLuint, 2>,
            HashPreSucVBOIdx, EqualPreSucVBOIdx> segs;

        friend class RenderPathManager;

    public:
        RenderPath(GLint id, const std::array<float, 3>& color)
            : id(id), color(color)
        {
            glGenVertexArrays(1, &vertVAO);
            glGenBuffers(1, &vertEBO);

            glBindVertexArray(vertVAO);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);

            // has bound global vertex VBO
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertEBO);
            glBufferStorage(GL_ELEMENT_ARRAY_BUFFER,
                sizeof(GLuint) * MAX_VERT_NUM, NULL, GL_DYNAMIC_STORAGE_BIT);

            glGenVertexArrays(1, &segVAO);
            glGenBuffers(1, &segEBO);

            glBindVertexArray(segVAO);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);

            // has bound global vertex VBO
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, segEBO);
            glBufferStorage(GL_ELEMENT_ARRAY_BUFFER,
                sizeof(GLuint) * 2 * MAX_VERT_NUM, NULL, GL_DYNAMIC_STORAGE_BIT);

            glBindVertexArray(0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        RenderPath(const RenderPath&) = delete;
        RenderPath(RenderPath&&) = delete;
        ~RenderPath()
        {
            if (vertVAO != 0) glDeleteVertexArrays(1, &vertVAO);
            if (vertEBO != 0) glDeleteBuffers(1, &vertEBO);
            if (segVAO != 0) glDeleteVertexArrays(1, &segVAO);
            if (segEBO != 0) glDeleteBuffers(1, &segEBO);
        }
        void addVertex(GLuint VBOIdx)
        {
            if (verts.find(VBOIdx) != verts.end()) return;

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertEBO);
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,
                sizeof(GLuint) * verts.size(), sizeof(GLuint), &VBOIdx);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            verts.insert(VBOIdx);
        }
        void deleteVertex(GLuint VBOIdx)
        {
            uint32_t linkNum = 0;
            std::array<GLuint, 2> needDelSeg;
            for (auto& seg : segs)
                if (seg[0] == VBOIdx || seg[1] == VBOIdx)
                {
                    needDelSeg = seg;
                    ++linkNum;
                }
            if (linkNum > 1)
                throw std::runtime_error(
                    "Vertex has multi-link, "
                    "cannot delete it. Delete Vertex from leaf to root");

            if (linkNum == 1) segs.erase(needDelSeg);
            verts.erase(VBOIdx);

            needUpdate = true;
        }
        void addSegment(const std::array<GLuint, 2>& preSucVBOIdx)
        {
            if (segs.find(preSucVBOIdx) != segs.end()) return;

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, segEBO);
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,
                sizeof(GLuint) * 2 * segs.size(), sizeof(GLuint) * 2,
                preSucVBOIdx.data());
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            segs.insert(preSucVBOIdx);
        }
        void spilitSegment(const std::array<GLuint, 2>& preSucVBOIdx, GLuint midVBOIdx)
        {
            auto itr = segs.find(preSucVBOIdx);
            if (itr == segs.end())
                throw std::runtime_error("2 selected Verteices are not adjacent");
            segs.erase(itr);

            verts.insert(midVBOIdx);
            segs.insert({ preSucVBOIdx[0], midVBOIdx });
            segs.insert({ preSucVBOIdx[1], midVBOIdx });

            needUpdate = true;
        }
        inline void draw()
        {
            if (needUpdate)
                update();

            glBindVertexArray(vertVAO);
            {
                glPointSize(POINT_SIZE);
                glDrawElements(GL_POINTS, verts.size(), GL_UNSIGNED_INT, 0);
                glPointSize(1.f);

                glLineWidth(LINE_WIDTH);
                glBindVertexArray(segVAO);
                glDrawElements(GL_LINES, 2 * segs.size(), GL_UNSIGNED_INT, 0);
                glLineWidth(1.f);
            }
            glBindVertexArray(0);
        }
        inline const auto& getSegments() const
        {
            return segs;
        }
        inline const auto& getVertices() const
        {
            return verts;
        }
        inline const auto& getColor() const
        {
            return color;
        }
        inline const int32_t getId() const
		{
			return id;
		}

	private:
		// According to verts, segs in CPU,
        // update all GPU Indices
		inline void update()
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertEBO);
			GLintptr EBOOffs = 0;
			for (auto VBOIdx : verts)
			{
				glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, EBOOffs, sizeof(GLuint), &VBOIdx);
				EBOOffs += sizeof(GLuint);
			}
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, segEBO);
			EBOOffs = 0;
			for (auto VBOIdx : segs)
			{
				glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, EBOOffs, sizeof(GLuint) * 2, VBOIdx.data());
				EBOOffs += sizeof(GLuint) * 2;
			}
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            needUpdate = false;
		}
    };

    class RenderPathManager
    {
    private:
        GLuint selectedPathID = std::numeric_limits<GLuint>::max();
        GLuint selectedVertID = std::numeric_limits<GLuint>::max();
        GLuint secondSelectedVertID = std::numeric_limits<GLuint>::max();
        GLuint VBO = 0;
        uint8_t availableColorIdx = 0;

        std::vector<std::array<float, 3>> availableColors;
        std::queue<GLuint> availablePathIDs;
        std::queue<GLuint> availableVBOIdxes;
        std::unordered_set<GLint> needUpdateRenderPaths;

        std::map<GLuint, RenderPath> renderPaths;
        std::unordered_map<GLuint, std::pair<GLint, glm::vec3>> VBOIdxToPathIDAndPos;

    public:
        RenderPathManager()
        {
            glGenBuffers(1, &VBO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferStorage(GL_ARRAY_BUFFER,
                sizeof(GLfloat) * MAX_VERT_NUM * 3, NULL, GL_DYNAMIC_STORAGE_BIT);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            for (GLuint VBOIdx = 0; VBOIdx < MAX_VERT_NUM; ++VBOIdx)
                availableVBOIdxes.push(VBOIdx);
            for (GLuint pathID = 0; pathID < MAX_VERT_NUM; ++pathID)
                availablePathIDs.push(pathID);

            availableColors.emplace_back(std::array{1.f, .5f, .5f});
            availableColors.emplace_back(std::array{.5f, 1.f, .5f});
            availableColors.emplace_back(std::array{.5f, .5f, 1.f});
        }
        RenderPathManager(const RenderPathManager&) = delete;
        ~RenderPathManager()
        {
            if (VBO != 0)
                glDeleteBuffers(1, &VBO);
        }
        inline void addPath()
        {
            if (availablePathIDs.empty())
                throw std::runtime_error("Reach MAX_PATH_NUM: "
                    + std::to_string(MAX_PATH_NUM));
            GLuint pathID = availablePathIDs.front();
            availablePathIDs.pop();

            auto& color = availableColors[availableColorIdx];
            ++availableColorIdx;
            if (availableColorIdx == availableColors.size())
                availableColorIdx = 0;

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            renderPaths.emplace(std::piecewise_construct,
                std::forward_as_tuple(pathID),
                std::forward_as_tuple(pathID, color));
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            selectedPathID = pathID;
            selectedVertID = secondSelectedVertID
                = std::numeric_limits<GLuint>::max();
        }
        inline void addVertex(const glm::vec3& pos)
        {
            if (selectedPathID == std::numeric_limits<GLuint>::max())
                throw std::runtime_error("No Path selected");
            if (renderPaths.at(selectedPathID).verts.size() != 0
                && selectedVertID == std::numeric_limits<GLuint>::max())
                throw std::runtime_error(
                    "The selected Path has Vertex, but NONE of them were selected. "
                    "Select 1 to ensure Path as a Tree struture");
            
            GLuint VBOIdx = allocVertex(pos);
            if (selectedVertID == std::numeric_limits<GLuint>::max())
                // only add vertex
                renderPaths.at(selectedPathID).addVertex(VBOIdx);
            else
            {
                // add vertex and segment
                renderPaths.at(selectedPathID).addVertex(VBOIdx);
                renderPaths.at(selectedPathID).addSegment({ selectedVertID, VBOIdx });
            }
            selectedVertID = VBOIdx;
        }
        inline void draw(
            GLuint shaderProgram,
            GLint shaderMatrixPos, const glm::mat4& mat,
            GLint shaderColorPos = -1)
        {
            glUseProgram(shaderProgram);
            glUniformMatrix4fv(shaderMatrixPos, 1, GL_FALSE, (GLfloat*)&mat);
            for (auto& [pathID, path] : renderPaths)
            {
                if (shaderColorPos != -1)
                    glUniform3fv(shaderColorPos, 1, (GLfloat*)path.color.data());
                path.draw();
            }
            glUseProgram(0);
        }

    private:
        inline GLuint allocVertex(const glm::vec3& pos)
        {
            if (availableVBOIdxes.empty())
                throw std::runtime_error(
                    "Reach MAX_VERT_NUM: " + std::to_string(MAX_VERT_NUM));
            GLuint VBOIdx = availableVBOIdxes.front();
            availableVBOIdxes.pop();
            VBOIdxToPathIDAndPos[VBOIdx] = std::make_pair(selectedPathID, pos);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            {
                auto& color = renderPaths.at(selectedPathID).color;
                std::array<float, 3> verts{ pos.x, pos.y, pos.z };
                glBufferSubData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 3 * VBOIdx,
                    sizeof(verts), verts.data());
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            return VBOIdx;
        }
    };
}

#endif // !KOUEK_RENDER_PATH
