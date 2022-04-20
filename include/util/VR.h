#ifndef KOUEK_VR_H
#define KOUEK_VR_H

#include <functional>

#include <glad/glad.h>
#include <openvr.h>

namespace
{
	class VRContext
	{
	public:
		enum class HandEnum : uint8_t
		{
			Left = 0,
			Right = 1
		};

        static inline std::function<void(std::function<void(uint8_t)>)> forHandsDo =
            [](std::function<void(uint8_t)> func) {
            for (uint8_t hndIdx = static_cast<uint8_t>(HandEnum::Left);
                hndIdx <= static_cast<uint8_t>(HandEnum::Right); ++hndIdx)
                func(hndIdx);
        };

        static inline constexpr uint8_t Hand_Left = static_cast<uint8_t>(HandEnum::Left);
        static inline constexpr uint8_t Hand_Right = static_cast<uint8_t>(HandEnum::Right);
		static inline std::function<void(std::function<void(uint8_t)>)> forEyesDo =
			[](std::function<void(uint8_t)> func) {
			for (uint8_t eyeIdx = vr::Eye_Left; eyeIdx <= vr::Eye_Right; ++eyeIdx)
				func(eyeIdx);
		};
	};

    class VRRenderModel
    {
    private:
        GLuint VBO = 0;
        GLuint EBO = 0;
        GLuint VAO = 0;
        GLuint tex = 0;
        GLsizei vertCnt = 0;

    public:
        VRRenderModel(
            const vr::RenderModel_t& vrModel,
            const vr::RenderModel_TextureMap_t& vrDiffuseTexture)
        {
            glGenVertexArrays(1, &VAO);
            glBindVertexArray(VAO);
            {
                glGenBuffers(1, &VBO);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(vr::RenderModel_Vertex_t) * vrModel.unVertexCount, vrModel.rVertexData,
                    GL_STATIC_DRAW);

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t),
                    (void*)offsetof(vr::RenderModel_Vertex_t, vPosition));
                glEnableVertexAttribArray(1);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t),
                    (void*)offsetof(vr::RenderModel_Vertex_t, vNormal));
                glEnableVertexAttribArray(2);
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t),
                    (void*)offsetof(vr::RenderModel_Vertex_t, rfTextureCoord));

                glGenBuffers(1, &EBO);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * vrModel.unTriangleCount * 3, vrModel.rIndexData,
                    GL_STATIC_DRAW);
            }
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vrDiffuseTexture.unWidth, vrDiffuseTexture.unHeight, 0, GL_RGBA,
                    GL_UNSIGNED_BYTE, vrDiffuseTexture.rubTextureMapData);

                glGenerateMipmap(GL_TEXTURE_2D);

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

                GLfloat largest;
                glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &largest);
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, largest);
            }
            glBindTexture(GL_TEXTURE_2D, 0);

            vertCnt = vrModel.unTriangleCount * 3;
        }
        VRRenderModel(const VRRenderModel&) = delete;
        ~VRRenderModel()
        {
            if (VAO != 0) glDeleteVertexArrays(1, &VAO);
            if (VBO != 0) glDeleteBuffers(1, &VBO);
            if (EBO != 0) glDeleteBuffers(1, &EBO);
            if (tex != 0) glDeleteTextures(1, &tex);
        }
        void draw()
        {
            glBindVertexArray(VAO);
            {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, tex);
                glDrawElements(GL_TRIANGLES, vertCnt, GL_UNSIGNED_SHORT, (const void*)0);
                glBindTexture(GL_TEXTURE_2D, 0);
            }
            glBindVertexArray(0);
        }
    };
}

#endif // !KOUEK_VR_H
