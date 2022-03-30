#ifndef KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
#define KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H

#include <renderer/Renderer.h>

#include <unordered_set>
#include <unordered_map>

#include <VolumeSlicer/volume_cache.hpp>

#include <Common/hash_function.hpp>
#include <Common/boundingbox.hpp>

namespace kouek
{
	// All CUDA funcs declaration
	namespace CompVolumeMonoEyeRendererImplCUDA
	{
		constexpr uint8_t MAX_LOD = 6;
		constexpr uint8_t MAX_TEX_UNIT_NUM = 10;

		struct CUDAParameter
		{
			glm::uvec3 texUnitDim;
		};
		struct CompVolumeParameter
		{
			uint32_t blockLength;
			uint32_t padding;
			uint32_t noPaddingBlockLength;
			glm::uvec3 LOD0BlockDim;
			glm::vec3 spaces;
		};
		struct RenderParameter
		{
			uint32_t maxStepNum;
			glm::uvec2 windowSize;
			float step;
			float projection22, projection23;
			float nearClip, farClip;
			glm::mat4 unProjection;
			glm::mat4 camRotaion;
			glm::vec3 camPos;
			glm::vec3 camFwd;
			CompVolumeRenderer::Subregion subrgn;
			CompVolumeRenderer::LightParamter lightParam;
		};

		void uploadCUDAParameter(const CUDAParameter* hostMemDat);
		void uploadBlockOffs(const uint32_t* hostMemDat, size_t num);
		void uploadCompVolumeParam(const CompVolumeParameter* hostMemDat);
		void uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num);
		void uploadTransferFunc(const float* hostMemDat);
		void uploadPreIntTransferFunc(const float* hostMemDat);
		void uploadRenderParam(const RenderParameter* hostMemDat);
		void uploadMappingTable(const uint32_t* hostMemDat, size_t size);

		void registerGLResource(GLuint outColorTex, GLuint inDepthTex, uint32_t w, uint32_t h);
		void unregisterGLResource();
		void render(uint32_t windowW, uint32_t windowH);
	}

	class CompVolumeMonoEyeRendererImpl : public CompVolumeMonoEyeRenderer
	{
	private:
		CUDAParameter cuda;
		bool subrgnChanged = true;
		CompVolumeMonoEyeRendererImplCUDA::CompVolumeParameter compVolumeParam;
		CompVolumeMonoEyeRendererImplCUDA::RenderParameter renderParam;

		std::shared_ptr<vs::CompVolume> volume;
		std::unique_ptr<vs::CUDAVolumeBlockCache> blockCache;

		std::unordered_map<std::array<uint32_t, 3>, vs::AABB, Hash_UInt32Array3> blockToAABBs;
		std::unordered_set<std::array<uint32_t, 4>, Hash_UInt32Array4> needBlocks, currNeedBlocks;
		std::unordered_set<std::array<uint32_t, 4>, Hash_UInt32Array4> loadBlocks, unloadBlocks;

	public:
		CompVolumeMonoEyeRendererImpl(const CUDAParameter& cudaParam);
		~CompVolumeMonoEyeRendererImpl();

		void registerGLResource(GLuint outColorTex, GLuint inDepthTex, uint32_t w, uint32_t h) override;
		void unregisterGLResource() override;
		void setStep(uint32_t maxStepNum, float step) override;
		void setSubregion(const Subregion& subrgn) override;
		void setTransferFunc(const vs::TransferFunc& tf) override;
		void setLightParam(const LightParamter& lightParam) override;
		void setVolume(std::shared_ptr<vs::CompVolume> volume) override;
		void render() override;

		void setCamera(const CameraParameter& camParam) override;
	};
}

#endif // !KOUEK_COMP_VOLUME_MONO_EYE_RENDERER_IMPL_H
