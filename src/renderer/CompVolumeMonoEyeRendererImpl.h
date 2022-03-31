#ifndef KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
#define KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H

#include "CompVolumeRendererImpl.h"

namespace kouek
{
	namespace CompVolumeRendererCUDA
	{
		class MonoEyeFunc : public Func
		{
		public:
			~MonoEyeFunc();

			virtual void uploadCompVolumeParam(const CompVolumeParameter& param) override ;
			virtual void uploadRenderParam(const RenderParameter& param) override ;
			virtual void uploadBlockOffs(const uint32_t* hostMemDat, size_t num) override ;
			virtual void uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num) override ;
			virtual void uploadTransferFunc(const float* hostMemDat) override ;
			virtual void uploadPreIntTransferFunc(const float* hostMemDat) override ;
			virtual void uploadMappingTable(const uint32_t* hostMemDat, size_t size) override ;

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
