#ifndef KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
#define KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H

#include "CompVolumeRendererImpl.h"

namespace kouek
{
	namespace CompVolumeRendererCUDA
	{
		constexpr uint8_t MAX_SUBSAMPLE_LEVEL_NUM = 6;
		constexpr float INTER_STAGE_OVERLAP_WIDTH_SQR = 1.f;

		struct FAVRRenderParameter : RenderParameter
		{
			glm::vec3 camPos2[2];
			glm::mat4 unProjection2[2];
			uint8_t sbsmplLvl;
		};

		class FAVRFunc : public Func
		{
		public:
			~FAVRFunc();

			void uploadCompVolumeParam(const CompVolumeParameter& param) override;
			void uploadBlockOffs(const uint32_t* hostMemDat, size_t num) override;
			void uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num) override;
			void uploadTransferFunc(const float* hostMemDat) override;
			void uploadPreIntTransferFunc(const float* hostMemDat) override;
			void uploadMappingTable(const uint32_t* hostMemDat, size_t size) override;

			void uploadRenderParam(const FAVRRenderParameter& param);
			void registerGLResource(
				GLuint outLftColorTex, GLuint outRhtColorTex,
				GLuint inLftDepthTex, GLuint inRhtDepthTex,
				uint32_t w, uint32_t h);
			void unregisterGLResource();
			void render(uint32_t windowW, uint32_t windowH, uint8_t sbsmplLvl);
		};
	}

	class CompVolumeFAVRRendererImpl :
		public CompVolumeRendererImpl,
		public CompVolumeFAVRRenderer
	{
	private:
		uint8_t subsampleLevel = 0;
		CompVolumeRendererCUDA::FAVRRenderParameter* FAVRRenderParam = nullptr;
		CompVolumeRendererCUDA::FAVRFunc* FAVRFunc = nullptr;

	public:
		CompVolumeFAVRRendererImpl(const CUDAParameter& cudaParam);
		~CompVolumeFAVRRendererImpl();

		void registerGLResource(
			GLuint outLftColorTex, GLuint outRhtColorTex,
			GLuint inLftDepthTex, GLuint inRhtDepthTex,
			uint32_t w, uint32_t h) override;
		void unregisterGLResource() override;

		void setCamera(const CameraParameter& camParam) override;
		void render() override;
	};
}

#endif // !KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
