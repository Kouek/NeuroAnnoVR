#ifndef KOUEK_SUBREGION_RENDERER_IMPL_H
#define KOUEK_SUBREGION_RENDERER_IMPL_H

#include <renderer/Renderer.h>

#include <unordered_set>

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

		struct CompVolumeParameter
		{
			uint32_t blockLength;
			uint32_t padding;
			uint32_t noPaddingBlockLength;
			uint3 LOD0BlockDim;
		};
		struct RenderParameter
		{
			uint32_t maxStepNum;
			uint2 windowSize;
			float maxStepDist;
			float3 spaces;
			glm::mat4 unProjection;
			glm::mat4 rotaion;
			glm::vec3 pos;
		};

		void uploadBlockOffs(const uint32_t* hostMemDat, size_t num);
		void uploadLightParam(const CompVolumeRenderer::LightParamter* hostMemDat);
		void uploadCompVolumeParam(const CompVolumeParameter* hostMemDat);
		void uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num);
		void uploadTransferFunc(const float* hostMemDat);
		void uploadPreIntTransferFunc(const float* hostMemDat);
		void uploadRenderParam(const RenderParameter* hostMemDat);
		void uploadMappingTable(const uint32_t* hostMemDat, size_t size);

		void registerOutputGLPBO(GLuint outPBO);
		void unregisterOutputGLPBO();
		void render(uint32_t windowW, uint32_t windowH);
	}

	class CompVolumeMonoEyeRendererImpl : public CompVolumeMonoEyeRenderer
	{
	private:
		CUDAParameter cuda;
		bool subrgnChanged = true;
		Subregion subrgn;
		CompVolumeMonoEyeRendererImplCUDA::CompVolumeParameter compVolumeParam;
		CompVolumeMonoEyeRendererImplCUDA::RenderParameter renderParam;

		std::shared_ptr<vs::CompVolume> volume;
		std::unique_ptr<vs::CUDAVolumeBlockCache> blockCache;

		std::unordered_set<std::array<uint32_t, 4>, Hash_UInt32Array4> needBlocks, currNeedBlocks;
		std::unordered_set<std::array<uint32_t, 4>, Hash_UInt32Array4> loadBlocks, unloadBlocks;

	public:
		CompVolumeMonoEyeRendererImpl(const CUDAParameter& cudaParam);
		~CompVolumeMonoEyeRendererImpl();

		void registerOutputGLPBO(GLuint outPBO, uint32_t w, uint32_t h) override;
		void unregisterOutputGLPBO() override;
		void setStep(uint32_t maxStepNum, float maxStepDist) override;
		void setSubregion(const Subregion& subrgn) override;
		void setTransferFunc(const vs::TransferFunc& tf) override;
		virtual void setLightParam(const LightParamter& lightParam) override;
		void setVolume(std::shared_ptr<vs::CompVolume> volume) override;
		void render() override;

		void setCamera(
			const glm::vec3& pos,
			const glm::mat3& rotation,
			const glm::mat4& unProjection) override;

	private:
		inline glm::vec3 float3ArrToGLMVec3(const std::array<float, 3> arr)
		{
			return glm::vec3(arr[0], arr[1], arr[2]);
		}
	};
}

#endif // !KOUEK_SUBREGION_RENDERER_IMPL_H
