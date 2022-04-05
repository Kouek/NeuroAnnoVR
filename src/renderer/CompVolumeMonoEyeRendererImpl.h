#ifndef KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
#define KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H

#include "CompVolumeRendererImpl.h"

namespace kouek
{
	namespace CompVolumeRendererCUDA
	{
		struct MonoEyeRenderParameter : RenderParameter
		{
			glm::vec3 camPos;
			__host__ __device__ MonoEyeRenderParameter() {}
		};

		class MonoEyeFunc : public Func
		{
		public:
			~MonoEyeFunc();

			void uploadCompVolumeParam(const CompVolumeParameter& param) override ;
			void uploadBlockOffs(const uint32_t* hostMemDat, size_t num) override ;
			void uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num) override ;
			void uploadTransferFunc(const float* hostMemDat) override ;
			void uploadPreIntTransferFunc(const float* hostMemDat) override ;
			void uploadMappingTable(const uint32_t* hostMemDat, size_t size) override ;

			void uploadRenderParam(const MonoEyeRenderParameter& param);
			void registerGLResource(GLuint outColorTex, GLuint inDepthTex, uint32_t w, uint32_t h);
			void unregisterGLResource();
			void render(uint32_t windowW, uint32_t windowH);
		};
	}

	class CompVolumeMonoEyeRendererImpl :
		public CompVolumeRendererImpl,
		public CompVolumeMonoEyeRenderer
	{
	private:
		CompVolumeRendererCUDA::MonoEyeRenderParameter* monoEyeRenderParam = nullptr;
		CompVolumeRendererCUDA::MonoEyeFunc* monoEyeFunc = nullptr;

	public:
		CompVolumeMonoEyeRendererImpl(const CUDAParameter& cudaParam);
		~CompVolumeMonoEyeRendererImpl();

		void registerGLResource(GLuint outColorTex, GLuint inDepthTex, uint32_t w, uint32_t h) override;
		void unregisterGLResource() override;

		void setCamera(const CameraParameter& camParam) override;
		void render() override;
	};
}

#endif // !KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
