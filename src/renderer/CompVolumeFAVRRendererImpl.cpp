#include "CompVolumeFAVRRendererImpl.h"

#include <Common/transfer_function_impl.hpp>

std::unique_ptr<kouek::CompVolumeFAVRRenderer>
kouek::CompVolumeFAVRRenderer::create(const CUDAParameter& cudaParam)
{
	return std::make_unique<CompVolumeFAVRRendererImpl>(cudaParam);
}

kouek::CompVolumeFAVRRendererImpl::CompVolumeFAVRRendererImpl(
	const CUDAParameter& cudaParam)
{
	this->cudaParm = cudaParam;

	renderParam = std::make_unique<CompVolumeRendererCUDA::FAVRRenderParameter>();
	FAVRRenderParam =
		static_cast<CompVolumeRendererCUDA::FAVRRenderParameter*>(renderParam.get());
	renderParam->texUnitDim = cudaParam.texUnitDim;

	cudaFunc = std::make_unique<CompVolumeRendererCUDA::FAVRFunc>();
	FAVRFunc = dynamic_cast<CompVolumeRendererCUDA::FAVRFunc*>(cudaFunc.get());
}

kouek::CompVolumeFAVRRendererImpl::~CompVolumeFAVRRendererImpl()
{

}

void kouek::CompVolumeFAVRRendererImpl::registerGLResource(
	GLuint outLftColorTex, GLuint outRhtColorTex,
	GLuint inLftDepthTex, GLuint inRhtDepthTex,
	uint32_t w, uint32_t h)
{
	renderParam->windowSize = { w,h };
	FAVRFunc->registerGLResource(outLftColorTex, outRhtColorTex,
		inLftDepthTex, inRhtDepthTex, w, h);
}

void kouek::CompVolumeFAVRRendererImpl::unregisterGLResource()
{
	FAVRFunc->unregisterGLResource();
}

void kouek::CompVolumeFAVRRendererImpl::setCamera(const CameraParameter& camParam)
{
	FAVRRenderParam->camPos2[0] = camParam.lftEyePos;
	FAVRRenderParam->camPos2[1] = camParam.rhtEyePos;
	FAVRRenderParam->unProjection2[0] = camParam.lftUnProjection;
	FAVRRenderParam->unProjection2[1] = camParam.rhtUnProjection;
	renderParam->camRotaion = camParam.rotation;
	renderParam->nearClip = camParam.nearClip;
	renderParam->farClip = camParam.farClip;

	// computed val
	renderParam->camFwd = glm::normalize(-renderParam->camRotaion[2]);
	renderParam->projection22 = -(renderParam->farClip + renderParam->nearClip) /
		(renderParam->farClip - renderParam->nearClip);
	renderParam->projection23 = -2.f * renderParam->farClip * renderParam->nearClip /
		(renderParam->farClip - renderParam->nearClip);
}

void kouek::CompVolumeFAVRRendererImpl::setInteractionParam(
	const InteractionParameter& intrctParam)
{
	FAVRRenderParam->intrctParam = intrctParam;
}

void kouek::CompVolumeFAVRRendererImpl::render(
	glm::vec3* intrctPos, RenderTarget renderTar)
{
	FAVRRenderParam->sbsmplLvl = 5;
	FAVRRenderParam->sbsmplSize.x = renderParam->windowSize.x / FAVRRenderParam->sbsmplLvl;
	FAVRRenderParam->sbsmplSize.y = FAVRRenderParam->sbsmplSize.x * FAVRRenderParam->sbsmplLvl;
	FAVRFunc->uploadRenderParam(*FAVRRenderParam);

	// filter blocks
	if (subrgnChanged)
	{
		loadBlocks.clear();
		unloadBlocks.clear();

		// according to Subregion, find blocks needed
		{
			vs::OBB subrgnOBB(
				renderParam->subrgn.center, renderParam->subrgn.rotation[0],
				renderParam->subrgn.rotation[1], renderParam->subrgn.rotation[2],
				renderParam->subrgn.halfW, renderParam->subrgn.halfH,
				renderParam->subrgn.halfD);
			// AABB filter first
			vs::AABB subrgnAABB = subrgnOBB.getAABB();
			for (auto& blockAABB : blockAABBs)
				if (subrgnAABB.intersect(blockAABB.second))
					currNeedBlocks.emplace(
						std::array{ blockAABB.first[0],
						blockAABB.first[1], blockAABB.first[2], (uint32_t)0 });
			// OBB filter then
			for (auto itr = currNeedBlocks.begin(); itr != currNeedBlocks.end();)
				if (!subrgnOBB.intersect_obb(
					blockAABBs[std::array{ (*itr)[0],(*itr)[1],(*itr)[2] }]
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
	cudaFunc->uploadMappingTable(
		mappingTable.data(), sizeof(uint32_t) * mappingTable.size());

	FAVRFunc->render(
		intrctPos,
		renderParam->windowSize.x, renderParam->windowSize.y,
		FAVRRenderParam->sbsmplSize.x, FAVRRenderParam->sbsmplSize.y,
		FAVRRenderParam->sbsmplLvl, renderTar);
}
