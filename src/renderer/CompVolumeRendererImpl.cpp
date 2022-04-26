#include "CompVolumeRendererImpl.h"

#include <Common/transfer_function_impl.hpp>

void kouek::CompVolumeRendererImpl::setStep(uint32_t maxStepNum, float step)
{
	renderParam->maxStepNum = maxStepNum;
	renderParam->step = step;
}

void kouek::CompVolumeRendererImpl::setSubregion(const Subregion& subrgn)
{
	renderParam->subrgn = subrgn;
	subrgnChanged = true;
}

void kouek::CompVolumeRendererImpl::setTransferFunc(const vs::TransferFunc& tf)
{
	vs::TransferFuncImpl tfImpl(tf);
	cudaFunc->uploadTransferFunc(
		tfImpl.getTransferFunction().data());
	cudaFunc->uploadPreIntTransferFunc(
		tfImpl.getPreIntTransferFunc().data());
}

void kouek::CompVolumeRendererImpl::setLightParam(const LightParamter& lightParam)
{
	renderParam->lightParam = lightParam;
}

void kouek::CompVolumeRendererImpl::setVolume(
	std::shared_ptr<vs::CompVolume> volume)
{
	this->volume = volume;

	blockCache = vs::CUDAVolumeBlockCache::Create(cudaParm.ctx);
	blockCache->SetCacheBlockLength(this->volume->GetBlockLength()[0]);
	blockCache->SetCacheCapacity(cudaParm.texUnitNum, cudaParm.texUnitDim.x,
		cudaParm.texUnitDim.y, cudaParm.texUnitDim.z);
	blockCache->CreateMappingTable(this->volume->GetBlockDim());

	{
		// map lod to flat({ lod,0,0,0 }),
		// which is the first Voxel idx of LOD lod
		auto& lodMappingTableOffsets = blockCache->GetLodMappingTableOffset();
		uint32_t maxLOD = 0, minLOD = std::numeric_limits<uint32_t>::max();
		for (auto& e : lodMappingTableOffsets)
		{
			if (e.first < minLOD) minLOD = e.first;
			if (e.first > maxLOD) maxLOD = e.first;
		}
		maxLOD--; // in lodMappingTableOffsets, Key ranges [0, MAX_LOD + 1]

		// map lod(idx of vector) to flat({ lod,0,0,0 }) / 4,
		// which is the first Block idx of LOD lod
		std::vector<uint32_t> blockOffsets((size_t)maxLOD + 1, 0);
		for (auto& e : lodMappingTableOffsets)
			// in lodMappingTableOffsets, Key ranges [0, MAX_LOD + 1],
			// while in blockOffsets, Key ranges [0, MAX_LOD]
			if (e.first <= maxLOD) blockOffsets[e.first] = e.second / 4;

		// upload to CUDA
		cudaFunc->uploadBlockOffs(
			blockOffsets.data(), blockOffsets.size());
	}

	{
		auto& blockLength = this->volume->GetBlockLength();
		compVolumeParam.blockLength = blockLength[0];
		compVolumeParam.padding = blockLength[1];
		compVolumeParam.noPaddingBlockLength = blockLength[0] - 2 * blockLength[1];
		auto& LOD0BlockDim = this->volume->GetBlockDim(0);
		compVolumeParam.LOD0BlockDim = glm::uvec3{
			LOD0BlockDim[0], LOD0BlockDim[1], LOD0BlockDim[2] };
		compVolumeParam.spaces = glm::vec3{
		volume->GetVolumeSpaceX(),
		volume->GetVolumeSpaceY(),
		volume->GetVolumeSpaceZ() };

		cudaFunc->uploadCompVolumeParam(compVolumeParam);
	}

	{
		auto& texObj = blockCache->GetCUDATextureObjects();
		cudaFunc->uploadCUDATextureObj(
			texObj.data(), texObj.size());
	}

	blockAABBs.clear(); // avoid conflict caused by Volume reset
	for (uint32_t z = 0; z < compVolumeParam.LOD0BlockDim.z; ++z)
		for (uint32_t y = 0; y < compVolumeParam.LOD0BlockDim.y; ++y)
			for (uint32_t x = 0; x < compVolumeParam.LOD0BlockDim.x; ++x)
				blockAABBs.emplace(
					std::piecewise_construct,
					std::forward_as_tuple(std::array{ x, y, z }),
					std::forward_as_tuple(
						glm::vec3{
							x * compVolumeParam.noPaddingBlockLength * compVolumeParam.spaces.x,
							y * compVolumeParam.noPaddingBlockLength * compVolumeParam.spaces.y,
							z * compVolumeParam.noPaddingBlockLength * compVolumeParam.spaces.z
						},
						glm::vec3{
							(x + 1) * compVolumeParam.noPaddingBlockLength * compVolumeParam.spaces.x,
							(y + 1) * compVolumeParam.noPaddingBlockLength * compVolumeParam.spaces.y,
							(z + 1) * compVolumeParam.noPaddingBlockLength * compVolumeParam.spaces.z
						},
						// dummy in this program
						std::array<uint32_t, 4>()
					)
				);
}
