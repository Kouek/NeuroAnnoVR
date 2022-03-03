#include "CompVolumeMonoEyeRendererImpl.h"

#include <Common/transfer_function_impl.hpp>

std::unique_ptr<kouek::CompVolumeMonoEyeRenderer>
kouek::CompVolumeMonoEyeRenderer::create(const CUDAParameter& cudaParam)
{
	return std::make_unique<CompVolumeMonoEyeRendererImpl>(cudaParam);
}

kouek::CompVolumeMonoEyeRendererImpl::CompVolumeMonoEyeRendererImpl(
	const CUDAParameter& cudaParam)
	: cuda(cudaParam)
{

}

kouek::CompVolumeMonoEyeRendererImpl::~CompVolumeMonoEyeRendererImpl()
{

}

void kouek::CompVolumeMonoEyeRendererImpl::registerOutputGLPBO(
	GLuint outPBO, uint32_t w, uint32_t h)
{
	renderParam.windowSize = make_uint2(w, h);
	CompVolumeMonoEyeRendererImplCUDA::registerOutputGLPBO(outPBO);
}

void kouek::CompVolumeMonoEyeRendererImpl::unregisterOutputGLPBO()
{
	CompVolumeMonoEyeRendererImplCUDA::unregisterOutputGLPBO();
}

void kouek::CompVolumeMonoEyeRendererImpl::setStep(uint32_t maxStepNum, float maxStepDist)
{
	renderParam.maxStepNum = maxStepNum;
	renderParam.maxStepDist = maxStepDist;
}

void kouek::CompVolumeMonoEyeRendererImpl::setSubregion(const Subregion& subrgn)
{
	renderParam.subrgn = subrgn;
}

void kouek::CompVolumeMonoEyeRendererImpl::setTransferFunc(const vs::TransferFunc& tf)
{
	vs::TransferFuncImpl tfImpl(tf);
	CompVolumeMonoEyeRendererImplCUDA::uploadTransferFunc(
		tfImpl.getTransferFunction().data());
	CompVolumeMonoEyeRendererImplCUDA::uploadPreIntTransferFunc(
		tfImpl.getPreIntTransferFunc().data());
}

void kouek::CompVolumeMonoEyeRendererImpl::setLightParam(const LightParamter& lightParam)
{
	renderParam.lightParam = lightParam;
}

void kouek::CompVolumeMonoEyeRendererImpl::setVolume(
	std::shared_ptr<vs::CompVolume> volume)
{
	this->volume = volume;

	blockCache = vs::CUDAVolumeBlockCache::Create(cuda.ctx);
	blockCache->SetCacheBlockLength(this->volume->GetBlockLength()[0]);
	blockCache->SetCacheCapacity(cuda.texUnitNum, cuda.texUnitDim[0], cuda.texUnitDim[1], cuda.texUnitDim[2]);
	blockCache->CreateMappingTable(this->volume->GetBlockDim());

	{
		// map lod to flat({ lod,0,0,0 }),
		// which is the first voxel idx of LOD lod
		auto& lodMappingTableOffsets = blockCache->GetLodMappingTableOffset();
		uint32_t maxLOD = 0, minLOD = std::numeric_limits<uint32_t>::max();
		for (auto& e : lodMappingTableOffsets)
		{
			if (e.first < minLOD) minLOD = e.first;
			if (e.first > maxLOD) maxLOD = e.first;
		}
		maxLOD--; // in lodMappingTableOffsets, Key ranges [0, MAX_LOD + 1]

		// map lod(idx of vector) to flat({ lod,0,0,0 }) / 4,
		// which is the first block idx of LOD lod
		std::vector<uint32_t> blockOffsets((size_t)maxLOD + 1, 0);
		for (auto& e : lodMappingTableOffsets)
			// in lodMappingTableOffsets, Key ranges [0, MAX_LOD + 1],
			// while in blockOffsets, Key ranges [0, MAX_LOD]
			if (e.first <= maxLOD) blockOffsets[e.first] = e.second / 4;

		// upload to CUDA
		CompVolumeMonoEyeRendererImplCUDA::uploadBlockOffs(
			blockOffsets.data(), blockOffsets.size());
	}

	{
		auto& blockLength = this->volume->GetBlockLength();
		compVolumeParam.blockLength = blockLength[0];
		compVolumeParam.padding = blockLength[1];
		compVolumeParam.noPaddingBlockLength = blockLength[0] - 2 * blockLength[1];
		auto& LOD0BlockDim = this->volume->GetBlockDim(0);
		compVolumeParam.LOD0BlockDim = make_uint3(LOD0BlockDim[0], LOD0BlockDim[1], LOD0BlockDim[2]);

		CompVolumeMonoEyeRendererImplCUDA::uploadCompVolumeParam(&compVolumeParam);
	}

	{
		auto& texObj = blockCache->GetCUDATextureObjects();
		CompVolumeMonoEyeRendererImplCUDA::uploadCUDATextureObj(
			texObj.data(), texObj.size());
	}
}

void kouek::CompVolumeMonoEyeRendererImpl::render()
{
	// spaces may be updated per frame,
	/*renderParam.spaces = make_float3(
		volume->GetVolumeSpaceX(), volume->GetVolumeSpaceY(), volume->GetVolumeSpaceZ());*/
	kouek::CompVolumeMonoEyeRendererImplCUDA::uploadRenderParam(&renderParam);

	//// filter blocks
	//if (subrgnChanged)
	//{
	//	loadBlocks.clear();
	//	unloadBlocks.clear();

	//	// according to Subregion, find blocks needed
	//	{
	//		std::array<float, 3> maxVoxel = {
	//			volume->GetVolumeDimX() - 1,
	//			volume->GetVolumeDimY() - 1,
	//			volume->GetVolumeDimZ() - 1,
	//		};
	//		
	//	}

	//	// loadBlocks = currNeedBlocks - (old)needBlocks
	//	for (auto& e : currNeedBlocks)
	//		if (needBlocks.find(e) == needBlocks.end()) loadBlocks.insert(e);
	//	// unloadBlocks = (old)needBlocks - currNeedBlocks
	//	for (auto& e : needBlocks)
	//		if (currNeedBlocks.find(e) == currNeedBlocks.end()) unloadBlocks.insert(e);

	//	needBlocks = std::move(currNeedBlocks);
	//	subrgnChanged = false;

	//	if (loadBlocks.size() > 0 || unloadBlocks.size() > 0)
	//	{
	//		// loadBlocks = loadBlocks - cachedBlocks
	//		decltype(loadBlocks) tmp;
	//		for (auto& e : loadBlocks)
	//		{
	//			bool cached = blockCache->SetCachedBlockValid(e);
	//			if (!cached) tmp.insert(e);
	//		}
	//		loadBlocks = std::move(tmp);

	//		for (auto& e : unloadBlocks)
	//			blockCache->SetBlockInvalid(e);

	//		volume->PauseLoadBlock();

	//		if (!needBlocks.empty())
	//		{
	//			std::vector<std::array<uint32_t, 4>> targets;
	//			targets.reserve(needBlocks.size());
	//			for (auto& e : needBlocks)
	//				targets.push_back(e);
	//			volume->ClearBlockInQueue(targets);
	//		}
	//		for (auto& e : loadBlocks)
	//			volume->SetRequestBlock(e);
	//		for (auto& e : unloadBlocks)
	//			volume->EraseBlockInRequest(e);

	//		volume->StartLoadBlock();
	//	}
	//}

	//for (auto& e : needBlocks)
	//{
	//	auto volumeBlock = volume->GetBlock(e);
	//	if (volumeBlock.valid)
	//	{
	//		blockCache->UploadVolumeBlock(e, volumeBlock.block_data->GetDataPtr(), volumeBlock.block_data->GetSize());
	//		volumeBlock.Release();
	//	}
	//}

	//auto& mappingTable = blockCache->GetMappingTable();
	//CompVolumeMonoEyeRendererImplCUDA::uploadMappingTable(
	//	mappingTable.data(), sizeof(uint32_t) * mappingTable.size());

	CompVolumeMonoEyeRendererImplCUDA::render(
		renderParam.windowSize.x, renderParam.windowSize.y);
}

void kouek::CompVolumeMonoEyeRendererImpl::setCamera(
	const glm::vec3& pos,
	const glm::mat4& rotation,
	const glm::mat4& unProjection)
{
	renderParam.camPos = pos;
	renderParam.camRotaion = rotation;
	renderParam.unProjection = unProjection;
}
