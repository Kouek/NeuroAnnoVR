#ifndef KOUEK_COMP_VOLUME_RENDERER_IMPL_H
#define KOUEK_COMP_VOLUME_RENDERER_IMPL_H
#include <renderer/Renderer.h>

#include <unordered_set>
#include <unordered_map>

#include <VolumeSlicer/volume_cache.hpp>

#include <Common/hash_function.hpp>
#include <Common/boundingbox.hpp>

namespace kouek
{
	namespace CompVolumeRendererCUDA
	{
		constexpr uint8_t MAX_LOD = 6;
		constexpr uint8_t MAX_TEX_UNIT_NUM = 10;

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
			glm::uvec3 texUnitDim;
			float step;
			float projection22, projection23;
			float nearClip, farClip;
			glm::mat4 unProjection;
			glm::mat4 camRotaion;
			glm::vec3 camFwd;
			CompVolumeRenderer::Subregion subrgn;
			CompVolumeRenderer::LightParamter lightParam;
		};

		class Func
		{
		public:
			virtual void uploadCompVolumeParam(const CompVolumeParameter& param) = 0;
			virtual void uploadBlockOffs(const uint32_t* hostMemDat, size_t num) = 0;
			virtual void uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num) = 0;
			virtual void uploadTransferFunc(const float* hostMemDat) = 0;
			virtual void uploadPreIntTransferFunc(const float* hostMemDat) = 0;
			virtual void uploadMappingTable(const uint32_t* hostMemDat, size_t size) = 0;
		};
	}

	class CompVolumeRendererImpl : virtual public CompVolumeRenderer
	{
	protected:
		CUDAParameter cudaParm; // should be initailized when constructed
		// Buffered Variables:
		//   not needed until start of rendering
		bool subrgnChanged = true;
		Subregion subrgn;
		CompVolumeRendererCUDA::CompVolumeParameter compVolumeParam;
		std::unique_ptr<CompVolumeRendererCUDA::RenderParameter> renderParam;
		std::unordered_map<std::array<uint32_t, 3>, vs::AABB, Hash_UInt32Array3> blockToAABBs;
		std::unordered_set<std::array<uint32_t, 4>, Hash_UInt32Array4> needBlocks, currNeedBlocks;
		std::unordered_set<std::array<uint32_t, 4>, Hash_UInt32Array4> loadBlocks, unloadBlocks;

		std::unique_ptr<CompVolumeRendererCUDA::Func> cudaFunc;
		std::shared_ptr<vs::CompVolume> volume;
		std::unique_ptr<vs::CUDAVolumeBlockCache> blockCache;

	public:
		void setStep(uint32_t maxStepNum, float step) override;
		void setSubregion(const Subregion& subrgn) override;
		void setTransferFunc(const vs::TransferFunc& tf) override;
		void setLightParam(const LightParamter& lightParam) override;
		void setVolume(std::shared_ptr<vs::CompVolume> volume) override;
	};
}

#endif // !KOUEK_COMP_VOLUME_RENDERER_IMPL_H
