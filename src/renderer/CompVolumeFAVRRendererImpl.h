#ifndef KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
#define KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H

#include "CompVolumeRendererImpl.h"

namespace kouek
{
	namespace CompVolumeRendererCUDA
	{
		constexpr uint8_t MAX_SUBSAMPLE_LEVEL_NUM = 6;
		constexpr uint8_t INTER_STAGE_OVERLAP_WIDTH = 10;
		constexpr uint32_t INTERACTION_SAMPLE_DIM = 100;
		constexpr float INTERACTION_SAMPLE_SCALAR_LOWER_THRESHOLD = .1f;

		struct FAVRRenderParameter : RenderParameter
		{
			glm::vec3 camPos2[2];
			glm::mat4 unProjection2[2];
			glm::uvec2 sbsmplSize;
			uint8_t sbsmplLvl;
			CompVolumeFAVRRenderer::InteractionParameter intrctParam;
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
			void render(
				glm::vec3* intrctPos,
				uint32_t windowW, uint32_t windowH,
				uint32_t sbsmplTexW, uint32_t sbsmplTexH,
				uint8_t sbsmplLvl, CompVolumeFAVRRenderer::RenderTarget renderTar);
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
		void setInteractionParam(const InteractionParameter intrctParam) override;
		void render() override
		{
			render(nullptr, static_cast<RenderTarget>(0));
		}
		void render(glm::vec3* intrctPos, RenderTarget renderTar) override;
	};
}

#endif // !KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
