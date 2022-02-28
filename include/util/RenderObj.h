#ifndef KOUEK_RENDER_OBJ_H
#define KOUEK_RENDER_OBJ_H

#include <glad/glad.h>

namespace kouek
{
    class WireFrame
    {
    protected:
        GLuint VBO = 0;
        GLuint VAO = 0;
        GLsizei vertCnt = 0;

    public:
        WireFrame(const WireFrame&) = delete;
        WireFrame(WireFrame&&) = delete;
        WireFrame& operator=(const WireFrame&) = delete;
        WireFrame& operator=(WireFrame&&) = delete;
        WireFrame(size_t vertCnt)
        {
            glGenVertexArrays(1, &VAO);
            glBindVertexArray(VAO);
            {
                glGenBuffers(1, &VBO);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertCnt * 6, NULL, GL_STATIC_DRAW);

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const void*)0);
                glEnableVertexAttribArray(1);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const void*)(sizeof(GLfloat) * 3));
            }
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            this->vertCnt = vertCnt;
        }
        void setSubData(const float* verts)
        {
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat) * vertCnt * 6, verts);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        WireFrame(const std::vector<GLfloat>& verts)
        {
            glGenVertexArrays(1, &VAO);
            glBindVertexArray(VAO);
            {
                glGenBuffers(1, &VBO);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * verts.size(), verts.data(), GL_STATIC_DRAW);

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const void*)0);
                glEnableVertexAttribArray(1);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const void*)(sizeof(GLfloat) * 3));
            }
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            vertCnt = verts.size() / 6;
        }
        ~WireFrame()
        {
            if (VAO != 0) glDeleteVertexArrays(1, &VAO);
            if (VBO != 0) glDeleteBuffers(1, &VBO);
        }
        inline void draw()
        {
            glBindVertexArray(VAO);
            glDrawArrays(GL_LINES, 0, vertCnt);
            glBindVertexArray(0);
        }
    };
}

#endif // !KOUEK_RENDER_OBJ_H
