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
	CompVolumeMonoEyeRendererImplCUDA::CUDAParameter param;
	param.texUnitDim = cuda.texUnitDim;
	CompVolumeMonoEyeRendererImplCUDA::uploadCUDAParameter(&param);
}

kouek::CompVolumeMonoEyeRendererImpl::~CompVolumeMonoEyeRendererImpl()
{

}

void kouek::CompVolumeMonoEyeRendererImpl::registerGLResource(
	GLuint outColorTex, GLuint inDepthTex, uint32_t w, uint32_t h)
{
	renderParam.windowSize = { w,h };
	CompVolumeMonoEyeRendererImplCUDA::registerGLResource(outColorTex, inDepthTex,
		w, h);
}

void kouek::CompVolumeMonoEyeRendererImpl::unregisterGLResource()
{
	CompVolumeMonoEyeRendererImplCUDA::unregisterGLResource();
}

void kouek::CompVolumeMonoEyeRendererImpl::setStep(uint32_t maxStepNum, float step)
{
	renderParam.maxStepNum = maxStepNum;
	renderParam.step = step;
}

void kouek::CompVolumeMonoEyeRendererImpl::setSubregion(const Subregion& subrgn)
{
	renderParam.subrgn = subrgn;
	subrgnChanged = true;
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
	blockCache->SetCacheCapacity(cuda.texUnitNum, cuda.texUnitDim.x, cuda.texUnitDim.y, cuda.texUnitDim.z);
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
		CompVolumeMonoEyeRendererImplCUDA::uploadBlockOffs(
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

		CompVolumeMonoEyeRendererImplCUDA::uploadCompVolumeParam(&compVolumeParam);
	}

	{
		auto& texObj = blockCache->GetCUDATextureObjects();
		CompVolumeMonoEyeRendererImplCUDA::uploadCUDATextureObj(
			texObj.data(), texObj.size());
	}

	blockToAABBs.clear(); // avoid conflict caused by Volume reset
	for (uint32_t z = 0; z < compVolumeParam.LOD0BlockDim.z; ++z)
		for (uint32_t y = 0; y < compVolumeParam.LOD0BlockDim.y; ++y)
			for (uint32_t x = 0; x < compVolumeParam.LOD0BlockDim.x; ++x)
				blockToAABBs.emplace(
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

void kouek::CompVolumeMonoEyeRendererImpl::render()
{
	kouek::CompVolumeMonoEyeRendererImplCUDA::uploadRenderParam(&renderParam);

	// filter blocks
	if (subrgnChanged)
	{
		loadBlocks.clear();
		unloadBlocks.clear();

		// according to Subregion, find blocks needed
		{
			vs::OBB subrgnOBB(
				renderParam.subrgn.center, renderParam.subrgn.rotation[0],
				renderParam.subrgn.rotation[1], renderParam.subrgn.rotation[2],
				renderParam.subrgn.halfW, renderParam.subrgn.halfH,
				renderParam.subrgn.halfD);
			// AABB filter first
			vs::AABB subrgnAABB = subrgnOBB.getAABB();
			for (auto& blockAABB : blockToAABBs)
				if (subrgnAABB.intersect(blockAABB.second))
					currNeedBlocks.emplace(
						std::array{ blockAABB.first[0],
						blockAABB.first[1], blockAABB.first[2], (uint32_t)0 });
			// OBB filter then
			for (auto itr = currNeedBlocks.begin(); itr != currNeedBlocks.end();)
				if (!subrgnOBB.intersect_obb(
					blockToAABBs[std::array{ (*itr)[0],(*itr)[1],(*itr)[2] }]
					.convertToOBB())
					)
					currNeedBlocks.erase(itr++);
				else
					++itr;
		}

		// loadBlocks = currNeedBlocks - (old)needBlocks
		for (auto& e : currNeedBlocks)
			if (needBlocks.find(e) == needBlocks.end()) loadBlocks.insert(e);
		// unloadBlocks = (old)needBlocks - currNeedBlocks
		for (auto& e : needBlocks)
			if (currNeedBlocks.find(e) == currNeedBlocks.end()) unloadBlocks.insert(e);

		needBlocks = std::move(currNeedBlocks);
		subrgnChanged = false;

		if (loadBlocks.size() > 0 || unloadBlocks.size() > 0)
		{
			// loadBlocks = loadBlocks - cachedBlocks
			decltype(loadBlocks) tmp;
			for (auto& e : loadBlocks)
			{
				bool cached = blockCache->SetCachedBlockValid(e);
				if (!cached) tmp.insert(e);
			}
			loadBlocks = std::move(tmp);

			for (auto& e : unloadBlocks)
				blockCache->SetBlockInvalid(e);

			volume->PauseLoadBlock();

			if (!needBlocks.empty())
			{
				std::vector<std::array<uint32_t, 4>> targets;
				targets.reserve(needBlocks.size());
				for (auto& e : needBlocks)
					targets.push_back(e);
				volume->ClearBlockInQueue(targets);
			}
			for (auto& e : loadBlocks)
				volume->SetRequestBlock(e);
			for (auto& e : unloadBlocks)
				volume->EraseBlockInRequest(e);

			volume->StartLoadBlock();
		}
	}

	for (auto& e : needBlocks)
	{
		auto volumeBlock = volume->GetBlock(e);
		if (volumeBlock.valid)
		{
			blockCache->UploadVolumeBlock(e, volumeBlock.block_data->GetDataPtr(), volumeBlock.block_data->GetSize());
			volumeBlock.Release();
		}
	}

	auto& mappingTable = blockCache->GetMappingTable();
	CompVolumeMonoEyeRendererImplCUDA::uploadMappingTable(
		mappingTable.data(), sizeof(uint32_t) * mappingTable.size());

	CompVolumeMonoEyeRendererImplCUDA::render(
		renderParam.windowSize.x, renderParam.windowSize.y);
}

void kouek::CompVolumeMonoEyeRendererImpl::setCamera(const CameraParameter& camParam)
{
	renderParam.camPos = camParam.pos;
	renderParam.camRotaion = camParam.rotation;
	renderParam.unProjection = camParam.unProjection;
	renderParam.projection22 = camParam.projection22;
	renderParam.projection23 = camParam.projection23;
	renderParam.nearClip = camParam.nearClip;
	renderParam.farClip = camParam.farClip;
}
