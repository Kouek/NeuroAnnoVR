#include "CompVolumeFAVRRendererImpl.h"

#include <Common/cuda_utils.hpp>

#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

using namespace kouek::CompVolumeRendererCUDA;

// CUDA Resource:
//   Allocated when needed,
//   freeed when CompVolumeRendererCUDA::FAVRFunc is deconstructed
__constant__ CompVolumeParameter dc_compVolumeParam;
__constant__ FAVRRenderParameter dc_renderParam;

__constant__ uint32_t dc_blockOffsets[MAX_LOD + 1];
__constant__ cudaTextureObject_t dc_textures[MAX_TEX_UNIT_NUM];

__constant__ cudaTextureObject_t dc_transferFunc;

cudaArray_t d_preIntTFArray = nullptr;
cudaTextureObject_t d_preIntTF;
__constant__ cudaTextureObject_t dc_preIntTransferFunc;

uint32_t* d_mappingTable = nullptr;
__constant__ size_t dc_mappingTableSize = 0;
__constant__ glm::uvec4* dc_mappingTableStride4 = nullptr;

static bool sbsmplLvlChanged = true;
cudaArray_t d_sbsmplTexArrs[MAX_SUBSAMPLE_LEVEL_NUM] = { nullptr };
cudaArray_t d_reconsTexArrs[MAX_SUBSAMPLE_LEVEL_NUM] = { nullptr };
cudaArray_t d_sbsmplColorTexArr2[2] = { nullptr };
cudaTextureObject_t d_sbsmplTexes[MAX_SUBSAMPLE_LEVEL_NUM];
cudaTextureObject_t d_reconsTexes[MAX_SUBSAMPLE_LEVEL_NUM];
cudaTextureObject_t d_sbsmplColorTex2[2];
__constant__ float dc_sbsmplRadSqrs[MAX_SUBSAMPLE_LEVEL_NUM + 1];
__constant__ cudaTextureObject_t dc_sbsmplTexes[MAX_SUBSAMPLE_LEVEL_NUM];
__constant__ cudaTextureObject_t dc_reconsTexes[MAX_SUBSAMPLE_LEVEL_NUM];
__constant__ cudaTextureObject_t dc_sbsmplColorTex2[2];

glm::u8vec4* d_color2[2] = { nullptr }, * d_sbsmplColor2[2] = { nullptr };
size_t d_colorSize, d_sbsmplColorSize;
cudaGraphicsResource_t outColorTexRsc2[2] = { nullptr };
cudaGraphicsResource_t inDepthTexRsc2[2] = { nullptr };
struct
{
	cudaResourceDesc rscDesc;
	cudaTextureDesc texDesc;
}depthTexDesc;
cudaTextureObject_t d_depthTex2[2];
cudaArray_t d_colorArr2[2] = { nullptr }, d_depthArr2[2] = { nullptr };

struct PosWithScalar
{
	glm::vec3 pos;
	float scalar;
};
PosWithScalar* d_intrctPossInXY = nullptr;
PosWithScalar* d_intrctPossInX = nullptr;
PosWithScalar* intrctPossInX = nullptr;
glm::vec3* d_intrctPos = nullptr;

cudaStream_t stream = nullptr;

kouek::CompVolumeRendererCUDA::FAVRFunc::~FAVRFunc()
{
	if (d_preIntTFArray != nullptr)
	{
		CUDA_RUNTIME_CHECK(
			cudaDestroyTextureObject(d_preIntTF));
		CUDA_RUNTIME_CHECK(
			cudaFreeArray(d_preIntTFArray));
		d_preIntTFArray = nullptr;
	}
	// TODO
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadCompVolumeParam(const CompVolumeParameter& param)
{
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_compVolumeParam, &param, sizeof(CompVolumeParameter)));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadRenderParam(
	const FAVRRenderParameter& param)
{
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_renderParam, &param, sizeof(FAVRRenderParameter)));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadBlockOffs(const uint32_t* hostMemDat, size_t num)
{
	assert(num <= MAX_LOD + 1);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_blockOffsets, hostMemDat, sizeof(uint32_t) * num));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num)
{
	assert(num <= MAX_TEX_UNIT_NUM);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_textures, hostMemDat, sizeof(cudaTextureObject_t) * num));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadTransferFunc(const float* hostMemDat)
{
	// TODO
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadPreIntTransferFunc(const float* hostMemDat)
{
	if (d_preIntTFArray == nullptr)
		CreateCUDATexture2D(256, 256, &d_preIntTFArray, &d_preIntTF);
	UpdateCUDATexture2D(
		(uint8_t*)hostMemDat, d_preIntTFArray, sizeof(float) * 256 * 4, 256, 0, 0);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_preIntTransferFunc, &d_preIntTF, sizeof(cudaTextureObject_t)));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadMappingTable(const uint32_t* hostMemDat, size_t size)
{
	if (d_mappingTable == nullptr)
	{
		cudaMalloc(&d_mappingTable, size);
		// cpy uint32_t ptr to uint4 ptr
		CUDA_RUNTIME_API_CALL(
			cudaMemcpyToSymbol(dc_mappingTableStride4, &d_mappingTable, sizeof(glm::uvec4*)));
	}
	CUDA_RUNTIME_API_CALL(
		cudaMemcpyToSymbol(dc_mappingTableSize, &size, sizeof(size_t)));
	CUDA_RUNTIME_API_CALL(
		cudaMemcpy(d_mappingTable, hostMemDat, size, cudaMemcpyHostToDevice));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::registerGLResource(
	GLuint outLftColorTex, GLuint outRhtColorTex,
	GLuint inLftDepthTex, GLuint inRhtDepthTex,
	uint32_t w, uint32_t h)
{
	d_colorSize = sizeof(glm::u8vec4) * w * h;
	CUDA_RUNTIME_API_CALL(
		cudaGraphicsGLRegisterImage(&outColorTexRsc2[0], outLftColorTex,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	CUDA_RUNTIME_API_CALL(
		cudaGraphicsGLRegisterImage(&outColorTexRsc2[1], outRhtColorTex,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	CUDA_RUNTIME_API_CALL(
		cudaMalloc(&d_color2[0], d_colorSize));
	CUDA_RUNTIME_API_CALL(
		cudaMalloc(&d_color2[1], d_colorSize));

	CUDA_RUNTIME_API_CALL(
		cudaGraphicsGLRegisterImage(&inDepthTexRsc2[0], inLftDepthTex,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	CUDA_RUNTIME_API_CALL(
		cudaGraphicsGLRegisterImage(&inDepthTexRsc2[1], inRhtDepthTex,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

	memset(&depthTexDesc.rscDesc, 0, sizeof(cudaResourceDesc));
	depthTexDesc.rscDesc.resType = cudaResourceTypeArray;
	memset(&depthTexDesc.texDesc, 0, sizeof(cudaTextureDesc));
	depthTexDesc.texDesc.normalizedCoords = 0;
	depthTexDesc.texDesc.filterMode = cudaFilterModePoint;
	depthTexDesc.texDesc.addressMode[0] = cudaAddressModeClamp;
	depthTexDesc.texDesc.addressMode[1] = cudaAddressModeClamp;
	depthTexDesc.texDesc.readMode = cudaReadModeElementType;
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::unregisterGLResource()
{
	for (uint8_t idx = 0; idx < 2; ++idx)
		if (outColorTexRsc2[idx] != nullptr)
		{
			CUDA_RUNTIME_API_CALL(cudaGraphicsUnregisterResource(outColorTexRsc2[idx]));
			outColorTexRsc2[idx] = nullptr;
			CUDA_RUNTIME_API_CALL(cudaFree(d_color2[idx]));
			d_color2[idx] = nullptr;
			CUDA_RUNTIME_API_CALL(cudaFree(d_sbsmplColor2[idx]));
			d_sbsmplColor2[idx] = nullptr;

			CUDA_RUNTIME_API_CALL(cudaGraphicsUnregisterResource(inDepthTexRsc2[idx]));
			inDepthTexRsc2[idx] = nullptr;
		}
}

static void createSubsampleColorTex(
	uint32_t sbsmplTexW, uint32_t sbsmplTexH)
{
	if (d_sbsmplColorTexArr2[0] != nullptr)
	{
		CUDA_RUNTIME_API_CALL(cudaFree(d_sbsmplColor2[0]));
		CUDA_RUNTIME_API_CALL(cudaFree(d_sbsmplColor2[1]));
		CUDA_RUNTIME_API_CALL(cudaDestroyTextureObject(d_sbsmplColorTex2[0]));
		CUDA_RUNTIME_API_CALL(cudaDestroyTextureObject(d_sbsmplColorTex2[1]));
		CUDA_RUNTIME_API_CALL(cudaFreeArray(d_sbsmplColorTexArr2[0]));
		CUDA_RUNTIME_API_CALL(cudaFreeArray(d_sbsmplColorTexArr2[1]));
		d_sbsmplColorTexArr2[0] = d_sbsmplColorTexArr2[1] = nullptr;
	}

	d_sbsmplColorSize = sizeof(glm::u8vec4) * sbsmplTexW * sbsmplTexH;
	CUDA_RUNTIME_API_CALL(
		cudaMalloc(&d_sbsmplColor2[0], d_sbsmplColorSize));
	CUDA_RUNTIME_API_CALL(
		cudaMalloc(&d_sbsmplColor2[1], d_sbsmplColorSize));
	{
		cudaChannelFormatDesc chnnlDesc = cudaCreateChannelDesc(
			8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_sbsmplColorTexArr2[0], &chnnlDesc,
				sbsmplTexW, sbsmplTexH));
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_sbsmplColorTexArr2[1], &chnnlDesc,
				sbsmplTexW, sbsmplTexH));
		cudaResourceDesc rscDesc;
		memset(&rscDesc, 0, sizeof(cudaResourceDesc));
		rscDesc.resType = cudaResourceTypeArray;
		rscDesc.res.array.array = d_sbsmplColorTexArr2[0];
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.normalizedCoords = 0;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.readMode = cudaReadModeNormalizedFloat;
		CUDA_RUNTIME_API_CALL(
			cudaCreateTextureObject(&d_sbsmplColorTex2[0], &rscDesc, &texDesc, nullptr));

		rscDesc.res.array.array = d_sbsmplColorTexArr2[1];
		CUDA_RUNTIME_API_CALL(
			cudaCreateTextureObject(&d_sbsmplColorTex2[1], &rscDesc, &texDesc, nullptr));

		CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(
			dc_sbsmplColorTex2, d_sbsmplColorTex2, sizeof(cudaTextureObject_t) * 2));
	}
}

__global__ void createSubsampleTexKernel(glm::vec4* d_sbsmpl)
{
	uint32_t texX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t texY = blockIdx.y * blockDim.y + threadIdx.y;
	if (texX >= dc_renderParam.sbsmplSize.x
		|| texY >= dc_renderParam.sbsmplSize.y) return;
	size_t texFlatIdx = (size_t)texY * dc_renderParam.sbsmplSize.x + texX;

	float centerY, hfSbsmplW;
	centerY = hfSbsmplW = .5f * dc_renderParam.sbsmplSize.x;
	float hfWindowWid = .5f * dc_renderParam.windowSize.x;
	float x, y, radSqr, scale, sbsmplX, sbsmplY;
	// 0 for no, 1 for having value in 8 near-regions,
	// 2 for having value itself
	int hasVal = 0;

	for (uint8_t stage = 0;
		stage < dc_renderParam.sbsmplLvl;
		++stage, centerY += dc_renderParam.sbsmplSize.x)
	{
		if (texY < centerY - hfSbsmplW || texY >= centerY + hfSbsmplW) continue;

		scale = 1.f - (float)stage / dc_renderParam.sbsmplLvl;
		x = (float)texX - hfSbsmplW;
		y = (float)texY - centerY;
		sbsmplX = x / scale + hfWindowWid;
		sbsmplY = y / scale + hfWindowWid;

		// id: <y_double, x_double, y_sign, x_sign, y, x>
		// 111011 101011 101010 101111 111111
		// 011011 001011 001010 001111 011111
		// 010001 000001 000000 000101 010101
		// 010011 000011 000010 000111 010111
		// 110011 100011 100010 100111 110111
		scale *= scale;
		float lowerBound = dc_sbsmplRadSqrs[stage] * scale;
		float upperBound = dc_sbsmplRadSqrs[stage + 1] * scale;
		for (uint8_t id = 0; id < 25; ++id)
		{
			x = (float)texX - hfSbsmplW;
			y = (float)texY - centerY;
			x += ((id & 0x10) ? 2.f : 1.f) * ((id & 0x4) ? 1.f : -1.f) * ((id & 0x1) ? 1.f : 0.f);
			y += ((id & 0x20) ? 2.f : 1.f) * ((id & 0x8) ? 1.f : -1.f) * ((id & 0x2) ? 1.f : 0.f);
			radSqr = x * x + y * y;
			if (radSqr >= lowerBound && radSqr < upperBound)
			{
				if (id == 0) hasVal = 2;
				else hasVal = 1;
				break;
			}
		}

		if (hasVal != 0)
		{
			d_sbsmpl[texFlatIdx].x = sbsmplX;
			d_sbsmpl[texFlatIdx].y = sbsmplY;
			d_sbsmpl[texFlatIdx].z = 0;
			d_sbsmpl[texFlatIdx].w = hasVal == 2 ? 1.f : .5f;
		}
		else
		{
			d_sbsmpl[texFlatIdx].x = d_sbsmpl[texFlatIdx].y
				= d_sbsmpl[texFlatIdx].z = 1.f;
			d_sbsmpl[texFlatIdx].w = 0;
		}
	}
}

__global__ void createReconstructionTexKernel(glm::vec4* d_recons)
{
	uint32_t texX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t texY = blockIdx.y * blockDim.y + threadIdx.y;
	if (texX >= dc_renderParam.windowSize.x || texY >= dc_renderParam.windowSize.x) return;
	size_t texFlatIdx = (size_t)texY * dc_renderParam.windowSize.x + texX;
	uint8_t idx = dc_renderParam.sbsmplLvl - 1;

	float hfW = .5f * dc_renderParam.windowSize.x;
	float hfSbsmplW = .5f * dc_renderParam.sbsmplSize.x;
	float sbsmplCntrY = .5f * dc_renderParam.sbsmplSize.x;
	float x, y, radSqr, scale;
	x = (float)texX - hfW;
	y = (float)texY - hfW;
	radSqr = x * x + y * y;
	glm::vec2 xy = { x,y };
	glm::vec2 drc = glm::normalize(xy);

	uint8_t stage;
	for (stage = 0; stage < dc_renderParam.sbsmplLvl;
		++stage, sbsmplCntrY += dc_renderParam.sbsmplSize.x)
	{
		if (radSqr < dc_sbsmplRadSqrs[stage] || radSqr >= dc_sbsmplRadSqrs[stage + 1])
			continue;

		scale = 1.f - (float)stage / dc_renderParam.sbsmplLvl;
		x = xy.x * scale + hfSbsmplW;
		y = xy.y * scale + sbsmplCntrY;
		break; // necessary, avoid sbsmplCntrY increasing
	}
	// deal with inter-stage gap
	float4 sbsmplTexVal = tex2D<float4>(
		dc_sbsmplTexes[idx], x, y);
	if (sbsmplTexVal.w != 1.f)
	{
		float maxRadSqr = .5f * dc_renderParam.windowSize.x;
		maxRadSqr *= maxRadSqr;
		float flag = (radSqr - dc_sbsmplRadSqrs[stage]) <
			(fminf(dc_sbsmplRadSqrs[stage + 1], maxRadSqr) - radSqr)
			? +1.f : -1.f;
		for (uint8_t wid = 0; wid < INTER_STAGE_OVERLAP_WIDTH; ++wid)
		{
			xy += flag * drc;

			x = xy.x * scale + hfSbsmplW;
			y = xy.y * scale + sbsmplCntrY;
			sbsmplTexVal = tex2D<float4>(
				dc_sbsmplTexes[idx], x, y);
			if (sbsmplTexVal.w == 1.f) break;
		}
	}
	d_recons[texFlatIdx].x = x;
	d_recons[texFlatIdx].y = y;
	d_recons[texFlatIdx].z = 0;
	d_recons[texFlatIdx].w = 1.f;
}

static void createSubsampleAndReconstructionTexes(
	uint8_t lvl, uint32_t w, uint32_t h,
	uint32_t sbsmplTexW, uint32_t sbsmplTexH)
{
	uint8_t idx = lvl - 1;
	{
		cudaChannelFormatDesc chnnlDesc = cudaCreateChannelDesc<float4>();
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_sbsmplTexArrs[idx], &chnnlDesc,
				sbsmplTexW, sbsmplTexH));
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_reconsTexArrs[idx], &chnnlDesc,
				w, w));
	}
	{
		cudaResourceDesc rscDesc;
		memset(&rscDesc, 0, sizeof(cudaResourceDesc));
		rscDesc.resType = cudaResourceTypeArray;
		rscDesc.res.array.array = d_sbsmplTexArrs[idx];
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.normalizedCoords = 0;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.readMode = cudaReadModeElementType;
		CUDA_RUNTIME_API_CALL(
			cudaCreateTextureObject(&d_sbsmplTexes[idx], &rscDesc, &texDesc, nullptr));
		CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(
			dc_sbsmplTexes, d_sbsmplTexes, sizeof(cudaTextureObject_t) * MAX_SUBSAMPLE_LEVEL_NUM));

		rscDesc.res.array.array = d_reconsTexArrs[idx];
		CUDA_RUNTIME_API_CALL(
			cudaCreateTextureObject(&d_reconsTexes[idx], &rscDesc, &texDesc, nullptr));
		CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(
			dc_reconsTexes, d_reconsTexes, sizeof(cudaTextureObject_t) * MAX_SUBSAMPLE_LEVEL_NUM));
	}
	{
		float sbsmplRadSqrs[MAX_SUBSAMPLE_LEVEL_NUM + 1] = { 0 };
		for (uint8_t stage = 1; stage < lvl; ++stage)
		{
			sbsmplRadSqrs[stage] = .5f * w / (lvl - stage + 1);
			sbsmplRadSqrs[stage] *= sbsmplRadSqrs[stage];
		}
		sbsmplRadSqrs[lvl] = INFINITY;
		CUDA_RUNTIME_API_CALL(
			cudaMemcpyToSymbol(dc_sbsmplRadSqrs, sbsmplRadSqrs,
				sizeof(float) * (MAX_SUBSAMPLE_LEVEL_NUM + 1)));
	}
	{
		glm::vec4* d_tmp = nullptr;
		size_t d_tmpSize = sizeof(glm::vec4) * sbsmplTexW * sbsmplTexH;
		CUDA_RUNTIME_API_CALL(cudaMalloc(&d_tmp, d_tmpSize));

		dim3 threadPerBlock = { 16, 16 };
		dim3 blockPerGrid = { (sbsmplTexW + threadPerBlock.x - 1) / threadPerBlock.x,
							 (sbsmplTexH + threadPerBlock.y - 1) / threadPerBlock.y };
		createSubsampleTexKernel << <blockPerGrid, threadPerBlock, 0, stream >> > (d_tmp);

		CUDA_RUNTIME_API_CALL(cudaMemcpyToArray(
			d_sbsmplTexArrs[idx], 0, 0, d_tmp, d_tmpSize, cudaMemcpyDeviceToDevice));
		CUDA_RUNTIME_API_CALL(cudaFree(d_tmp));

		d_tmpSize = sizeof(glm::vec4) * w * w;
		CUDA_RUNTIME_API_CALL(cudaMalloc(&d_tmp, d_tmpSize));
		blockPerGrid = { (w + threadPerBlock.x - 1) / threadPerBlock.x,
							 (w + threadPerBlock.y - 1) / threadPerBlock.y };
		createReconstructionTexKernel << <blockPerGrid, threadPerBlock, 0, stream >> > (d_tmp);

		CUDA_RUNTIME_API_CALL(cudaMemcpyToArray(
			d_reconsTexArrs[idx], 0, 0, d_tmp, d_tmpSize, cudaMemcpyDeviceToDevice));
		CUDA_RUNTIME_API_CALL(cudaFree(d_tmp));
	}
}

__device__ float virtualSampleLOD0(const glm::vec3& samplePos)
{
	// sample pos in Voxel Space -> virtual sample Block idx
	glm::uvec3 vsBlockIdx =
		samplePos / (float)dc_compVolumeParam.noPaddingBlockLength;

	// virtual sample Block idx -> real sample Block idx (in GPU Mem)
	glm::uvec4 GPUMemBlockIdx;
	{
		size_t flatVSBlockIdx = dc_blockOffsets[0]
			+ vsBlockIdx.z * dc_compVolumeParam.LOD0BlockDim.y * dc_compVolumeParam.LOD0BlockDim.x
			+ vsBlockIdx.y * dc_compVolumeParam.LOD0BlockDim.x
			+ vsBlockIdx.x;
		if (flatVSBlockIdx >= dc_mappingTableSize) return 0;
		GPUMemBlockIdx = dc_mappingTableStride4[flatVSBlockIdx];
	}

	if (((GPUMemBlockIdx.w >> 16) & (0x0000ffff)) != 1)
		// not a valid GPU Mem block
		return 0;

	// sample pos in Voxel Space -> real sample pos (in GPU Mem)
	glm::vec3 GPUMemSamplePos;
	{
		glm::vec3 offsetInNoPaddingBlock = samplePos -
			glm::vec3{ vsBlockIdx * dc_compVolumeParam.noPaddingBlockLength };
		GPUMemSamplePos = glm::vec3{ GPUMemBlockIdx.x, GPUMemBlockIdx.y, GPUMemBlockIdx.z }
			*(float)dc_compVolumeParam.blockLength
			+ offsetInNoPaddingBlock + (float)dc_compVolumeParam.padding;
		// normolized
		GPUMemSamplePos /= dc_renderParam.texUnitDim;
	}

	return tex3D<float>(dc_textures[GPUMemBlockIdx.w & (0x0000ffff)],
		GPUMemSamplePos.x, GPUMemSamplePos.y, GPUMemSamplePos.z);
}

__global__ void findInteractionPosInXYZKernel(PosWithScalar* d_intrctPossInXY)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= INTERACTION_SAMPLE_DIM || y >= INTERACTION_SAMPLE_DIM) return;
	size_t flatIdx = y * INTERACTION_SAMPLE_DIM + x;

	float maxScalar = 0;
	glm::vec3 samplePos, intrctPos;
	glm::vec3 subrgnCenterInWdSp = {
		dc_renderParam.subrgn.halfW,
		dc_renderParam.subrgn.halfH,
		dc_renderParam.subrgn.halfD,
	};
	glm::vec3 step3 = dc_renderParam.intrctParam.dat.ball.AABBSize
		/ (float)INTERACTION_SAMPLE_DIM;
	glm::vec3 pos = dc_renderParam.intrctParam.dat.ball.startPos;
	pos.x += step3.x * x;
	pos.y += step3.y * y;
	for (uint32_t stepCnt = 0;
		stepCnt < INTERACTION_SAMPLE_DIM;
		++stepCnt, pos.z += step3.z)
	{
		// ray pos in World Space -> sample pos in Voxel Space
		samplePos =
			(pos - subrgnCenterInWdSp + dc_renderParam.subrgn.center)
			/ dc_compVolumeParam.spaces;
		float scalar = virtualSampleLOD0(samplePos);
		if (scalar > maxScalar)
		{
			maxScalar = scalar;
			intrctPos = pos;
		}
	}
	d_intrctPossInXY[flatIdx].pos = intrctPos;
	d_intrctPossInXY[flatIdx].scalar = maxScalar;
}

__global__ void findInteractionPosInXYKernel(
	PosWithScalar* d_intrctPossInX, PosWithScalar* d_intrctPossInXY)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= INTERACTION_SAMPLE_DIM) return;

	float maxScalar = 0;
	glm::vec3 intrctPos;
	size_t flatIdx = x;
	for (uint32_t y = 0;
		y < INTERACTION_SAMPLE_DIM;
		++y, flatIdx += INTERACTION_SAMPLE_DIM)
	{
		float scalar = d_intrctPossInXY[flatIdx].scalar;
		if (scalar > maxScalar)
		{
			maxScalar = scalar;
			intrctPos = d_intrctPossInXY[flatIdx].pos;
		}
	}
	d_intrctPossInX[x].pos = intrctPos;
	d_intrctPossInX[x].scalar = maxScalar;
}

__device__ glm::u8vec4 rgbaFloatToUbyte4(float r, float g, float b, float a)
{
	r = __saturatef(r); // clamp to [0.0, 1.0]
	g = __saturatef(g);
	b = __saturatef(b);
	a = __saturatef(a);
	r *= 255.f;
	g *= 255.f;
	b *= 255.f;
	a *= 255.f;
	return glm::u8vec4(r, g, b, a);
}

__device__ void rayIntersectAABB(
	float* tEnter, float* tExit,
	const glm::vec3& rayOri, const glm::vec3& rayDrc,
	const glm::vec3& bot, const glm::vec3& top)
{
	// For  Ori + Drc * t3Bot.x = <Bot.x, 0, 0>
	// Thus t3Bot.x = Bot.x / Drc.x
	// Thus t3Bot.y = Bot.x / Drc.y
	// If  \
	//  \_\|\ 
	//   \_\|
	//      \.t3Bot.x
	//      |\
	//    __|_\.___|
	//      |  \t3Bot.y
	//    __|___\._|_
	//    t3Top.y\ |
	//      |     \.t3Top.x
	// 
	// Then t3Min = t3Bot, t3Max = t3Top
	// And  the max of t3Min is tEnter
	// And  the min of t3Max is tExit

	glm::vec3 invRay = 1.f / rayDrc;
	glm::vec3 t3Bot = invRay * (bot - rayOri);
	glm::vec3 t3Top = invRay * (top - rayOri);
	glm::vec3 t3Min{
		fminf(t3Bot.x, t3Top.x),
		fminf(t3Bot.y, t3Top.y),
		fminf(t3Bot.z, t3Top.z) };
	glm::vec3 t3Max{
		fmaxf(t3Bot.x, t3Top.x),
		fmaxf(t3Bot.y, t3Top.y),
		fmaxf(t3Bot.z, t3Top.z) };
	*tEnter = fmaxf(fmaxf(t3Min.x, t3Min.y), fmaxf(t3Min.x, t3Min.z));
	*tExit = fminf(fminf(t3Max.x, t3Max.y), fminf(t3Max.x, t3Max.z));
}

__device__ glm::vec3 phongShadingLOD0(
	const glm::vec3& rayDrc, const glm::vec3& samplePos,
	const glm::vec3& diffuseColor)
{
	glm::vec3 N;
	{
		float val1, val2;
		val1 = virtualSampleLOD0(samplePos + glm::vec3{ 1.f,0,0 });
		val2 = virtualSampleLOD0(samplePos - glm::vec3{ 1.f,0,0 });
		N.x = val2 - val1;
		val1 = virtualSampleLOD0(samplePos + glm::vec3{ 0,1.f,0 });
		val2 = virtualSampleLOD0(samplePos - glm::vec3{ 0,1.f,0 });
		N.y = val2 - val1;
		val1 = virtualSampleLOD0(samplePos + glm::vec3{ 0,0,1.f });
		val2 = virtualSampleLOD0(samplePos - glm::vec3{ 0,0,1.f });
		N.z = val2 - val1;
	}
	N = glm::normalize(N);

	glm::vec3 L = { -rayDrc.x,-rayDrc.y,-rayDrc.z };
	glm::vec3 R = L;
	if (glm::dot(N, L) < 0) N = -N;

	glm::vec3 ambient = dc_renderParam.lightParam.ka * diffuseColor;
	glm::vec3 specular = glm::vec3(dc_renderParam.lightParam.ks
		* powf(fmaxf(dot(N, .5f * (L + R)), 0),
			dc_renderParam.lightParam.shininess));
	glm::vec3 diffuse = dc_renderParam.lightParam.kd
		* fmaxf(dot(N, L), 0.f) * diffuseColor;

	return ambient + specular + diffuse;
}

// WARNING:
// - Declaring type of param d_depth as [const cudaTextureObject_t &]
//   will cause unknown error at tex2D()
__global__ void renderKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR,
	cudaTextureObject_t d_depthTexL, cudaTextureObject_t d_depthTexR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.windowSize.x || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	// render Left or Right Eye
	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	d_color = rgbaFloatToUbyte4(
		dc_renderParam.lightParam.bkgrndColor.r,
		dc_renderParam.lightParam.bkgrndColor.g,
		dc_renderParam.lightParam.bkgrndColor.b,
		dc_renderParam.lightParam.bkgrndColor.a);

#ifdef TEST_DEPTH_TEX
	uchar4 depth4 = tex2D<uchar4>(
		blockIdx.z == 0 ? d_depthTexL : d_depthTexR,
		windowX, windowY);
	float depInZeroOne = depth4.x / 255.f;
	d_color = rgbaFloatToUbyte4(
		depInZeroOne, depInZeroOne, depInZeroOne, 1.f);
	return;
#endif // TEST_DEPTH_TEX


	glm::vec3 rayDrc;
	const glm::vec3& camPos = dc_renderParam.camPos2[blockIdx.z];
	float tEnter, tExit;
	{
		// find Ray of each Pixel on Window
		//   unproject
		const glm::mat4& unProjection = dc_renderParam.unProjection2[blockIdx.z];
		glm::vec4 v41 = unProjection * glm::vec4{
			(((float)windowX / dc_renderParam.windowSize.x) - .5f) * 2.f,
			(((float)windowY / dc_renderParam.windowSize.y) - .5f) * 2.f,
			1.f, 1.f };
		//   don't rotate first to compute the Near&Far-clip steps
		rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;
		rayDrc = glm::normalize(rayDrc);
		float absRayDrcZ = fabsf(rayDrc.z);
		float tNearClip = dc_renderParam.nearClip / absRayDrcZ;
		float tFarClip = dc_renderParam.farClip;
		//   then compute upper bound of steps
		//   for Mesh-Volume mixed rendering
		{
			uchar4 depth4 = tex2D<uchar4>(
				blockIdx.z == 0 ? d_depthTexL : d_depthTexR,
				windowX, windowY);
			float meshBoundDep = dc_renderParam.projection23 /
				((depth4.x / 255.f * 2.f - 1.f) + dc_renderParam.projection22);
			if (tFarClip > meshBoundDep)
				tFarClip = meshBoundDep;
		} 
		tFarClip /= absRayDrcZ;
		//   rotate
		v41.x = rayDrc.x, v41.y = rayDrc.y, v41.z = rayDrc.z; // normalized in vec3
		v41 = dc_renderParam.camRotaion * v41;
		rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;

		// Ray intersect Subregion(OBB)
		// equivalent to Ray intersect AABB in Subreion Space
		//   for pos, apply Rotation and Translation
		glm::vec4 v42{ camPos.x, camPos.y, camPos.z, 1.f };
		v42 = dc_renderParam.subrgn.fromWorldToSubrgn * v42;
		//   for drc, apply Rotation only
		v41.w = 0;
		v41 = dc_renderParam.subrgn.fromWorldToSubrgn * v41;
		rayIntersectAABB(
			&tEnter, &tExit,
			glm::vec3(v42),
			glm::normalize(glm::vec3(v41)),
			glm::zero<glm::vec3>(),
			glm::vec3{
				dc_renderParam.subrgn.halfW * 2,
				dc_renderParam.subrgn.halfH * 2,
				dc_renderParam.subrgn.halfD * 2 });

		// Near&Far-clip
		if (tEnter < tNearClip) tEnter = tNearClip;
		if (tExit > tFarClip) tExit = tFarClip;
	}

	// no intersection
	if (tEnter >= tExit)
		return;
	glm::vec3 rayPos = camPos + tEnter * rayDrc;

#ifdef TEST_RAY_ENTER_POSITION
	// TEST: Ray Enter Position
	d_color = rgbaFloatToUbyte4(
		.5f * rayPos.x / d_renderParam.subrgn.halfW,
		.5f * rayPos.y / d_renderParam.subrgn.halfH,
		.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
	return;
#endif // TEST_RAY_ENTER_POSITION

#ifdef TEST_RAY_EXIT_POSITION
	// TEST: Ray Exit Position
	rayPos = camPos + tExit * rayDrc;
	d_color = rgbaFloatToUbyte4(
		.5f * rayPos.x / d_renderParam.subrgn.halfW,
		.5f * rayPos.y / d_renderParam.subrgn.halfH,
		.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
	return;
#endif // TEST_RAY_EXIT_POSITION

	glm::vec3 subrgnCenterInWdSp = {
		.5f * dc_renderParam.subrgn.halfW,
		.5f * dc_renderParam.subrgn.halfH,
		.5f * dc_renderParam.subrgn.halfD,
	};
	glm::vec3 rayDrcMulStp = rayDrc * dc_renderParam.step;
	glm::vec3 samplePos;
	glm::vec4 color = glm::zero<glm::vec4>();
	float sampleVal = 0;
	uint32_t stepNum = 0;
	for (;
		stepNum <= dc_renderParam.maxStepNum && tEnter <= tExit;
		++stepNum, tEnter += dc_renderParam.step, rayPos += rayDrcMulStp)
	{
		// ray pos in World Space -> sample pos in Voxel Space
		samplePos =
			(rayPos - subrgnCenterInWdSp + dc_renderParam.subrgn.center)
			/ dc_compVolumeParam.spaces;

		// virtual sample in Voxel Space, real sample in GPU Mem
		float currSampleVal = virtualSampleLOD0(samplePos);
		if (currSampleVal <= 0)
			continue;

		float4 currColor = tex2D<float4>(dc_preIntTransferFunc, sampleVal, currSampleVal);
		if (currColor.w <= 0)
			continue;

		glm::vec3 shadingColor = phongShadingLOD0(rayDrc,
			samplePos, glm::vec3{ currColor.x,currColor.y,currColor.z });
		currColor.x = shadingColor.x;
		currColor.y = shadingColor.y;
		currColor.z = shadingColor.z;

		sampleVal = currSampleVal;
		color = color + (1.f - color.w) * glm::vec4{ currColor.x,currColor.y,currColor.z,currColor.w }
		*glm::vec4{ currColor.w,currColor.w,currColor.w,1.f };

		if (color.w > 0.9f)
			break;
	}

	// gamma correction
	constexpr float GAMMA_CORRECT_COEF = 1.f / 2.2f;
	color.r = powf(color.r, GAMMA_CORRECT_COEF);
	color.g = powf(color.g, GAMMA_CORRECT_COEF);
	color.b = powf(color.b, GAMMA_CORRECT_COEF);

	d_color = rgbaFloatToUbyte4(color.r, color.g, color.b, color.a);
}

__global__ void testSubsampleTexKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.sbsmplSize.x
		|| windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float4 sbsmplTexVal = tex2D<float4>(dc_sbsmplTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX, (float)windowY);
	d_color = rgbaFloatToUbyte4(
		sbsmplTexVal.x / dc_renderParam.windowSize.x,
		sbsmplTexVal.y / dc_renderParam.windowSize.x,
		sbsmplTexVal.z, 1.f);
}

__global__ void subsampleKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR, glm::vec3* d_intrctPos,
	cudaTextureObject_t d_depthTexL, cudaTextureObject_t d_depthTexR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.sbsmplSize.x
		|| windowY >= dc_renderParam.sbsmplSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.sbsmplSize.x + windowX;
	
	// render Left or Right Eye
	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	d_color = rgbaFloatToUbyte4(
		dc_renderParam.lightParam.bkgrndColor.r,
		dc_renderParam.lightParam.bkgrndColor.g,
		dc_renderParam.lightParam.bkgrndColor.b,
		dc_renderParam.lightParam.bkgrndColor.a);

	// use (0,0) to find the interaction position when using AnnotationLaser
	bool isThreadForFindIntrctPos = dc_renderParam.intrctParam.mode ==
		kouek::CompVolumeFAVRRenderer::InteractionMode::AnnotationLaser
		&& windowX == 0 && windowY == 0;	

	// sample the original Window Pos
	float4 sbsmplTexVal = tex2D<float4>(
		dc_sbsmplTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX, (float)windowY);
	if (isThreadForFindIntrctPos); // no filtering here
	else if (sbsmplTexVal.w == 0)
	{
		d_color = rgbaFloatToUbyte4(0, 0, 1.f, 1.f);
		return;
	}
	else if (sbsmplTexVal.x >= dc_renderParam.windowSize.x
		|| sbsmplTexVal.y >= dc_renderParam.windowSize.y) return;

#ifdef TEST_DEPTH_TEX
	uchar4 depth4 = tex2D<uchar4>(
		blockIdx.z == 0 ? d_depthTexL : d_depthTexR,
		sbsmplTexVal.x, sbsmplTexVal.y);
	float depInZeroOne = depth4.x / 255.f;
	d_color = rgbaFloatToUbyte4(
		depInZeroOne, depInZeroOne, depInZeroOne, 1.f);
	return;
#endif // TEST_DEPTH_TEX

	glm::vec3 rayDrc;
	const glm::vec3& camPos = isThreadForFindIntrctPos ?
		dc_renderParam.intrctParam.dat.laser.ori : dc_renderParam.camPos2[blockIdx.z];
	float tEnter, tExit;
	{
		// find Ray of each Pixel on Window
		//   unproject
		const glm::mat4& unProjection = dc_renderParam.unProjection2[blockIdx.z];
		glm::vec4 v41 = isThreadForFindIntrctPos ?
			glm::vec4{ dc_renderParam.intrctParam.dat.laser.drc, 1.f } :
			unProjection * glm::vec4{
			((sbsmplTexVal.x / dc_renderParam.windowSize.x) - .5f) * 2.f,
			((sbsmplTexVal.y / dc_renderParam.windowSize.y) - .5f) * 2.f,
			1.f, 1.f };
		//   don't rotate first to compute the Near&Far-clip steps
		rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;
		rayDrc = glm::normalize(rayDrc);
		float absRayDrcZ = fabsf(rayDrc.z);
		float tNearClip = dc_renderParam.nearClip / absRayDrcZ;
		float tFarClip = dc_renderParam.farClip;
		//   then compute upper bound of steps
		//   for Mesh-Volume mixed rendering
		if (!isThreadForFindIntrctPos)
		{
			uchar4 depth4 = tex2D<uchar4>(
				blockIdx.z == 0 ? d_depthTexL : d_depthTexR,
				sbsmplTexVal.x, sbsmplTexVal.y);
			float meshBoundDep = dc_renderParam.projection23 /
				((depth4.x / 255.f * 2.f - 1.f) + dc_renderParam.projection22);
			if (tFarClip > meshBoundDep)
				tFarClip = meshBoundDep;
		}
		tFarClip /= absRayDrcZ;
		//   rotate
		v41.x = rayDrc.x, v41.y = rayDrc.y, v41.z = rayDrc.z; // normalized in vec3
		if (!isThreadForFindIntrctPos)
		{
			v41 = dc_renderParam.camRotaion * v41;
			rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;
		}

		// Ray intersect Subregion(OBB)
		// equivalent to Ray intersect AABB in Subreion Space
		//   for pos, apply Rotation and Translation
		glm::vec4 v42{ camPos.x, camPos.y, camPos.z, 1.f };
		v42 = dc_renderParam.subrgn.fromWorldToSubrgn * v42;
		//   for drc, apply Rotation only
		v41.w = 0;
		v41 = dc_renderParam.subrgn.fromWorldToSubrgn * v41;
		rayIntersectAABB(
			&tEnter, &tExit,
			glm::vec3(v42),
			glm::normalize(glm::vec3(v41)),
			glm::zero<glm::vec3>(),
			glm::vec3{
				dc_renderParam.subrgn.halfW * 2,
				dc_renderParam.subrgn.halfH * 2,
				dc_renderParam.subrgn.halfD * 2 });

		// Near&Far-clip
		if (tEnter < tNearClip) tEnter = tNearClip;
		if (tExit > tFarClip) tExit = tFarClip;

		// no intersection
		if (tEnter >= tExit)
			return;
	}

	glm::vec3 rayPos = camPos + tEnter * rayDrc;
	if (isThreadForFindIntrctPos) *d_intrctPos = rayPos;

	glm::vec3 subrgnCenterInWdSp = {
		dc_renderParam.subrgn.halfW,
		dc_renderParam.subrgn.halfH,
		dc_renderParam.subrgn.halfD,
	};
	glm::vec3 rayDrcMulStp = rayDrc * dc_renderParam.step;
	glm::vec3 samplePos;
	glm::vec4 color = glm::zero<glm::vec4>();
	float sampleVal = 0, maxScalar = 0;
	uint32_t stepNum = 0;
	for (;
		stepNum <= dc_renderParam.maxStepNum && tEnter <= tExit;
		++stepNum, tEnter += dc_renderParam.step, rayPos += rayDrcMulStp)
	{
		// ray pos in World Space -> sample pos in Voxel Space
		samplePos =
			(rayPos - subrgnCenterInWdSp + dc_renderParam.subrgn.center)
			/ dc_compVolumeParam.spaces;

		// virtual sample in Voxel Space, real sample in GPU Mem
		float currSampleVal = virtualSampleLOD0(samplePos);
		if (currSampleVal <= 0)
			continue;
		if (isThreadForFindIntrctPos && currSampleVal > maxScalar)
		{
			maxScalar = currSampleVal;
			*d_intrctPos = rayPos;
		}

		float4 currColor = tex2D<float4>(dc_preIntTransferFunc, sampleVal, currSampleVal);
		if (currColor.w <= 0)
			continue;

		glm::vec3 shadingColor = phongShadingLOD0(rayDrc,
			samplePos, glm::vec3{ currColor.x,currColor.y,currColor.z });
		currColor.x = shadingColor.x;
		currColor.y = shadingColor.y;
		currColor.z = shadingColor.z;

		sampleVal = currSampleVal;
		color = color + (1.f - color.w) * glm::vec4{ currColor.x,currColor.y,currColor.z,currColor.w }
		*glm::vec4{ currColor.w,currColor.w,currColor.w,1.f };

		if (color.w > 0.9f)
			break;
	}

	// gamma correction
	constexpr float GAMMA_CORRECT_COEF = 1.f / 2.2f;
	color.r = powf(color.r, GAMMA_CORRECT_COEF);
	color.g = powf(color.g, GAMMA_CORRECT_COEF);
	color.b = powf(color.b, GAMMA_CORRECT_COEF);

	d_color = rgbaFloatToUbyte4(color.r, color.g, color.b, color.a);
}

__global__ void testSubsampleKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.sbsmplSize.x
		|| windowY >= dc_renderParam.sbsmplSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float4 sbsmplVal = tex2D<float4>(
		dc_sbsmplColorTex2[blockIdx.z],
		(float)windowX, (float)windowY);
	d_color = rgbaFloatToUbyte4(sbsmplVal.x, sbsmplVal.y, sbsmplVal.z, 1.f);
}

__global__ void testReconstructionTexKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.windowSize.x || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float4 reconsTexVal = tex2D<float4>(
		dc_reconsTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX, (float)windowY);
	d_color = rgbaFloatToUbyte4(
		reconsTexVal.x / dc_renderParam.sbsmplSize.x,
		reconsTexVal.y / dc_renderParam.sbsmplSize.y,
		reconsTexVal.z, 1.f);
}

__global__ void testReconstructionXDiffKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.windowSize.x || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float4 reconsTexVal = tex2D<float4>(
		dc_reconsTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX, (float)windowY);
	float4 sbsmplTexVal = tex2D<float4>(
		dc_sbsmplTexes[dc_renderParam.sbsmplLvl - 1],
		reconsTexVal.x, reconsTexVal.y);
	/*if (sbsmplTexVal.w == 1.f)
		d_color = rgbaFloatToUbyte4(1.f, 1.f, 1.f, 1.f);
	else if (sbsmplTexVal.w == 0)
		d_color = rgbaFloatToUbyte4(0, 0, 0, 1.f);
	else
		d_color = rgbaFloatToUbyte4(1.f, .5f, 0, 1.f);*/
	float diff = sbsmplTexVal.x - (float)windowX;
	d_color = rgbaFloatToUbyte4(diff < 0 ? 0 : 1.f,
		fabsf(diff) > 0 ? 1.f : 0, 0, 1.f);
}

__global__ void testReconstructionYDiffKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.windowSize.x || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float4 reconsTexVal = tex2D<float4>(
		dc_reconsTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX, (float)windowY);
	float4 sbsmplTexVal = tex2D<float4>(
		dc_sbsmplTexes[dc_renderParam.sbsmplLvl - 1],
		reconsTexVal.x, reconsTexVal.y);
	/*if (sbsmplTexVal.w == 1.f)
		d_color = rgbaFloatToUbyte4(1.f, 1.f, 1.f, 1.f);
	else if (sbsmplTexVal.w == 0)
		d_color = rgbaFloatToUbyte4(0, 0, 0, 1.f);
	else
		d_color = rgbaFloatToUbyte4(1.f, .5f, 0, 1.f);*/
	float diff = sbsmplTexVal.y - (float)windowY;
	d_color = rgbaFloatToUbyte4(diff < 0 ? 0 : 1.f,
		0, fabsf(diff) > 0 ? 1.f : 0, 1.f);
}

__global__ void reconstructionKernel(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.windowSize.x || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float4 reconsTexVal = tex2D<float4>(
		dc_reconsTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX, (float)windowY);
	float4 sbsmplVal = tex2D<float4>(
		dc_sbsmplColorTex2[blockIdx.z],
		reconsTexVal.x, reconsTexVal.y);
	d_color = rgbaFloatToUbyte4(sbsmplVal.x, sbsmplVal.y, sbsmplVal.z, sbsmplVal.w);
}

//#define NO_FAVR
void kouek::CompVolumeRendererCUDA::FAVRFunc::render(
	glm::vec3* intrctPos, const CompVolumeFAVRRenderer::InteractionParameter& intrctParam,
	uint32_t windowW, uint32_t windowH,
	uint32_t sbsmplTexW, uint32_t sbsmplTexH,
	uint8_t sbsmplLvl, CompVolumeFAVRRenderer::RenderTarget renderTar)
{
	if (stream == nullptr)
		CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

	assert(sbsmplLvl > 0 && sbsmplLvl <= MAX_SUBSAMPLE_LEVEL_NUM);
	if (d_sbsmplTexArrs[sbsmplLvl - 1] == nullptr)
		createSubsampleAndReconstructionTexes(
			sbsmplLvl, windowW, windowH, sbsmplTexW, sbsmplTexH);
	if (sbsmplLvlChanged)
	{
		createSubsampleColorTex(sbsmplTexW, sbsmplTexH);
		sbsmplLvlChanged = false;
	}

	// from here, called per frame, thus no CUDA_RUNTIME_API_CHECK
	dim3 threadPerBlock = { 16, 16 };
	dim3 blockPerGrid;

	if (intrctParam.mode == CompVolumeFAVRRenderer::InteractionMode::AnnotationBall)
	{
		if (d_intrctPossInXY == nullptr)
		{
			CUDA_RUNTIME_API_CALL(
				cudaMalloc(&d_intrctPossInXY, sizeof(PosWithScalar)
					* INTERACTION_SAMPLE_DIM * INTERACTION_SAMPLE_DIM));
			CUDA_RUNTIME_API_CALL(
				cudaMalloc(&d_intrctPossInX, sizeof(PosWithScalar) * INTERACTION_SAMPLE_DIM));
			intrctPossInX = new PosWithScalar[INTERACTION_SAMPLE_DIM];
		}
		blockPerGrid = { (INTERACTION_SAMPLE_DIM + threadPerBlock.x - 1) / threadPerBlock.x,
						 (INTERACTION_SAMPLE_DIM + threadPerBlock.y - 1) / threadPerBlock.y };
		findInteractionPosInXYZKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_intrctPossInXY);

		blockPerGrid = { (INTERACTION_SAMPLE_DIM + threadPerBlock.x - 1) / threadPerBlock.x,
			1 };
		findInteractionPosInXYKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_intrctPossInX, d_intrctPossInXY);
		cudaMemcpy(intrctPossInX, d_intrctPossInX,
			sizeof(PosWithScalar) * INTERACTION_SAMPLE_DIM, cudaMemcpyDeviceToHost);
		float maxScalar = 0;
		if (intrctPos != nullptr)
			for (uint32_t x = 0; x < INTERACTION_SAMPLE_DIM; ++x)
				if (intrctPossInX[x].scalar > maxScalar
					&& intrctPossInX[x].scalar > INTERACTION_SAMPLE_SCALAR_LOWER_THRESHOLD)
				{
					maxScalar = intrctPossInX[x].scalar;
					*intrctPos = intrctPossInX[x].pos;
				}
	}
	else if (intrctParam.mode == CompVolumeFAVRRenderer::InteractionMode::AnnotationLaser)
		if (d_intrctPos == nullptr)
			CUDA_RUNTIME_API_CALL(
				cudaMalloc(&d_intrctPos, sizeof(glm::vec3)));

	for (uint8_t idx = 0; idx < 2; ++idx)
	{
		cudaGraphicsMapResources(1, &outColorTexRsc2[idx], stream);
		cudaGraphicsSubResourceGetMappedArray(&d_colorArr2[idx], outColorTexRsc2[idx], 0, 0);

		cudaGraphicsMapResources(1, &inDepthTexRsc2[idx], stream);
		cudaGraphicsSubResourceGetMappedArray(&d_depthArr2[idx], inDepthTexRsc2[idx], 0, 0);
		depthTexDesc.rscDesc.res.array.array = d_depthArr2[idx];
		cudaCreateTextureObject(&d_depthTex2[idx], &depthTexDesc.rscDesc,
			&depthTexDesc.texDesc, nullptr);
	}

	switch (renderTar)
	{
	case CompVolumeFAVRRenderer::RenderTarget::SubsampleTex:
		blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		testSubsampleTexKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_color2[0], d_color2[1]);

		break;
	case CompVolumeFAVRRenderer::RenderTarget::SubsampleResult:
		blockPerGrid = { (sbsmplTexW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (sbsmplTexH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		subsampleKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_sbsmplColor2[0], d_sbsmplColor2[1], intrctPos,
			d_depthTex2[0], d_depthTex2[1]);
		cudaMemcpyToArray(d_sbsmplColorTexArr2[0], 0, 0, d_sbsmplColor2[0],
			d_sbsmplColorSize, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(d_sbsmplColorTexArr2[1], 0, 0, d_sbsmplColor2[1],
			d_sbsmplColorSize, cudaMemcpyDeviceToDevice);

		blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
							 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		testSubsampleKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_color2[0], d_color2[1]);

		break;
	case CompVolumeFAVRRenderer::RenderTarget::ReconstructionTex:
		blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
							 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		testReconstructionTexKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_color2[0], d_color2[1]);

		break;
	case CompVolumeFAVRRenderer::RenderTarget::ReconstructionXDiff:
		blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
							 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		testReconstructionXDiffKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_color2[0], d_color2[1]);

		break;
	case CompVolumeFAVRRenderer::RenderTarget::ReconstructionYDiff:
		blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
							 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		testReconstructionYDiffKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_color2[0], d_color2[1]);

		break;
	case CompVolumeFAVRRenderer::RenderTarget::Image:
		blockPerGrid = { (sbsmplTexW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (sbsmplTexH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		subsampleKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_sbsmplColor2[0], d_sbsmplColor2[1], d_intrctPos,
			d_depthTex2[0], d_depthTex2[1]);
		if (intrctPos != nullptr
			&& intrctParam.mode == CompVolumeFAVRRenderer::InteractionMode::AnnotationLaser)
			cudaMemcpy(intrctPos, d_intrctPos, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
		cudaMemcpyToArray(d_sbsmplColorTexArr2[0], 0, 0, d_sbsmplColor2[0],
			d_sbsmplColorSize, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(d_sbsmplColorTexArr2[1], 0, 0, d_sbsmplColor2[1],
			d_sbsmplColorSize, cudaMemcpyDeviceToDevice);

		blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
							 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		reconstructionKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_color2[0], d_color2[1]);

		break;
	case CompVolumeFAVRRenderer::RenderTarget::FullResolutionImage:
		blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
		renderKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
			d_color2[0], d_color2[1], d_depthTex2[0], d_depthTex2[1]);

		break;
	default:
		assert(false);
	}

	for (uint8_t idx = 0; idx < 2; ++idx)
	{
		cudaMemcpyToArray(d_colorArr2[idx], 0, 0,
			d_color2[idx], d_colorSize, cudaMemcpyDeviceToDevice);

		d_colorArr2[idx] = d_depthArr2[idx] = nullptr;
		cudaGraphicsUnmapResources(1, &outColorTexRsc2[idx], stream);
		cudaGraphicsUnmapResources(1, &inDepthTexRsc2[idx], stream);
	}
}
