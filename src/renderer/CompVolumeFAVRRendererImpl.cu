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
__constant__ glm::uvec4* d_mappingTableStride4 = nullptr;

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
size_t d_colorSize;
cudaGraphicsResource_t outColorTexRsc2[2] = { nullptr };
cudaGraphicsResource_t inDepthTexRsc2[2] = { nullptr };
struct
{
	cudaResourceDesc rscDesc;
	cudaTextureDesc texDesc;
}depthTexDesc;
cudaTextureObject_t d_depthTex2[2];
cudaArray_t d_colorArr2[2] = { nullptr }, d_depthArr2[2] = { nullptr };

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
			cudaMemcpyToSymbol(d_mappingTableStride4, &d_mappingTable, sizeof(glm::uvec4*)));
	}
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

	CUDA_RUNTIME_API_CALL(
		cudaMalloc(&d_sbsmplColor2[0], d_colorSize));
	CUDA_RUNTIME_API_CALL(
		cudaMalloc(&d_sbsmplColor2[1], d_colorSize));
	{
		cudaChannelFormatDesc chnnlDesc = cudaCreateChannelDesc(
			8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_sbsmplColorTexArr2[0], &chnnlDesc, w, h));
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_sbsmplColorTexArr2[1], &chnnlDesc, w, h));
		cudaResourceDesc rscDesc;
		memset(&rscDesc, 0, sizeof(cudaResourceDesc));
		rscDesc.resType = cudaResourceTypeArray;
		rscDesc.res.array.array = d_sbsmplColorTexArr2[0];
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.normalizedCoords = 1;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
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

__global__ void createSubsampleTexKernel(
	glm::vec4* d_sbsmpl,
	uint32_t sbsmplTexW, uint32_t sbsmplTexH)
{
	uint32_t texX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t texY = blockIdx.y * blockDim.y + threadIdx.y;
	size_t texFlatIdx = (size_t)texY * sbsmplTexW + texX;
	if (texX > sbsmplTexW|| texY > sbsmplTexH) return;

	float centerY = .5f * sbsmplTexW;
	float hfSbsmplW = .5f * sbsmplTexW;
	float x, y, radSqr, scale;
	x = (float)texX - hfSbsmplW, y = (float)texY - centerY;
	radSqr = x * x + y * y;

	glm::vec4 X_Y_Null_HasVal;
	for (uint8_t stage = 0;
		stage < dc_renderParam.sbsmplLvl;
		++stage, centerY += sbsmplTexW)
	{
		if (texY < centerY - hfSbsmplW || texY >= centerY + hfSbsmplW) continue;

		scale = 1.f - (float)stage / dc_renderParam.sbsmplLvl;
		x /= sbsmplTexW, y /= sbsmplTexW;
		x /= scale, y /= scale;
		x = (x + 1.f) * .5f, y = (y + 1.f) * .5f;

		scale *= scale;

		if (radSqr >= dc_sbsmplRadSqrs[stage] * scale
			&& radSqr < dc_sbsmplRadSqrs[stage + 1] * scale)
		{
			X_Y_Null_HasVal.x = x;
			X_Y_Null_HasVal.y = y;
			X_Y_Null_HasVal.z = 0;
			X_Y_Null_HasVal.w = 1.f;
		}
		else
			X_Y_Null_HasVal.x = x = X_Y_Null_HasVal.y = y =
			X_Y_Null_HasVal.z = X_Y_Null_HasVal.w = 0;
		d_sbsmpl[texFlatIdx] = X_Y_Null_HasVal;
	}
}

__global__ void createReconstructionTexKernel(
	glm::vec4* d_recons, uint32_t sbsmplTexW, uint32_t sbsmplTexH)
{
	uint32_t texX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t texY = blockIdx.y * blockDim.y + threadIdx.y;
	size_t texFlatIdx = (size_t)texY * sbsmplTexW + texX;
	if (texX > sbsmplTexW || texY > sbsmplTexW) return;

	float sbsmplCntrY = .5f * sbsmplTexW;
	float hfSbsmplW = .5f * sbsmplTexW;
	float x, y, radSqr, scale;
	x = (float)texX - hfSbsmplW, y = (float)texY - hfSbsmplW;
	radSqr = x * x + y * y;

	for (uint8_t stage = 0;
		stage < dc_renderParam.sbsmplLvl;
		++stage, sbsmplCntrY += sbsmplTexW)
	{
		scale = 1.f - (float)stage / dc_renderParam.sbsmplLvl;
		scale *= scale;

		if (radSqr < dc_sbsmplRadSqrs[stage] * scale + stage == 0 ? 0 : INTER_STAGE_OVERLAP_WIDTH_SQR
			|| radSqr > dc_sbsmplRadSqrs[stage + 1] * scale + INTER_STAGE_OVERLAP_WIDTH_SQR)
			continue;

		x += hfSbsmplW, y += sbsmplCntrY;
		x /= sbsmplTexW, y /= sbsmplTexH;

		d_recons[texFlatIdx].x = x, d_recons[texFlatIdx].y = y,
			d_recons[texFlatIdx].z = 0, d_recons[texFlatIdx].w = 1.f;
	}
}

static void createSubsampleAndReconstructionTexes(uint8_t lvl, uint32_t w, uint32_t h)
{
	uint8_t idx = lvl - 1;
	uint32_t sbsmplTexW = w / lvl;
	uint32_t sbsmplTexH = sbsmplTexW * lvl;
	{
		cudaChannelFormatDesc chnnlDesc = cudaCreateChannelDesc<float4>();
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_sbsmplTexArrs[idx], &chnnlDesc,
				sbsmplTexW, sbsmplTexH));
		CUDA_RUNTIME_API_CALL(
			cudaMallocArray(&d_reconsTexArrs[idx], &chnnlDesc,
				sbsmplTexW, sbsmplTexW));
	}
	{
		cudaResourceDesc rscDesc;
		memset(&rscDesc, 0, sizeof(cudaResourceDesc));
		rscDesc.resType = cudaResourceTypeArray;
		rscDesc.res.array.array = d_sbsmplTexArrs[idx];
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.normalizedCoords = 1;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
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
			sbsmplRadSqrs[stage] = (float)sbsmplTexW / (lvl - stage + 1);
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
		createSubsampleTexKernel << <blockPerGrid, threadPerBlock, 0, stream >> > (
			d_tmp, sbsmplTexW, sbsmplTexH);

		CUDA_RUNTIME_API_CALL(cudaMemcpyToArray(
			d_sbsmplTexArrs[idx], 0, 0, d_tmp, d_tmpSize, cudaMemcpyDeviceToDevice));

		d_tmpSize = sizeof(glm::vec4) * sbsmplTexW * sbsmplTexW;
		blockPerGrid = { (sbsmplTexW + threadPerBlock.x - 1) / threadPerBlock.x,
							 (sbsmplTexW + threadPerBlock.y - 1) / threadPerBlock.y };
		createReconstructionTexKernel << <blockPerGrid, threadPerBlock, 0, stream >> > (
			d_tmp, sbsmplTexW, sbsmplTexH);

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
		GPUMemBlockIdx = d_mappingTableStride4[flatVSBlockIdx];
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

	glm::vec3 rayDrc;
	const glm::vec3& camPos = dc_renderParam.camPos2[blockIdx.z];
	const glm::mat4 unProjection = dc_renderParam.unProjection2[blockIdx.z];
	float tEnter, tExit;
	{
		// find Ray of each Pixel on Window
		//   unproject
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
	uint32_t windowW = dc_renderParam.windowSize.x / dc_renderParam.sbsmplLvl;
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= windowW || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float4 sbsmplTexVal = tex2D<float4>(dc_sbsmplTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX / windowW, (float)windowY / dc_renderParam.windowSize.y);
	d_color = rgbaFloatToUbyte4(sbsmplTexVal.x, sbsmplTexVal.y, sbsmplTexVal.z, 1.f);
}

__global__ void FAVRSubsample(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR,
	cudaTextureObject_t d_depthTexL, cudaTextureObject_t d_depthTexR)
{
	uint32_t windowW = dc_renderParam.windowSize.x / dc_renderParam.sbsmplLvl;
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= windowW || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;
	
	// render Left or Right Eye
	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	d_color = rgbaFloatToUbyte4(
		dc_renderParam.lightParam.bkgrndColor.r,
		dc_renderParam.lightParam.bkgrndColor.g,
		dc_renderParam.lightParam.bkgrndColor.b,
		dc_renderParam.lightParam.bkgrndColor.a);

	// sample the original Window Pos
	glm::vec2 oriWindow;
	float4 sbsmplTexVal = tex2D<float4>(dc_sbsmplTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX / windowW, (float)windowY / dc_renderParam.windowSize.y);
	if (sbsmplTexVal.w == 0) return;
	oriWindow.x = sbsmplTexVal.x * dc_renderParam.windowSize.x;
	oriWindow.y = sbsmplTexVal.y * dc_renderParam.windowSize.y;
	if (oriWindow.x >= dc_renderParam.windowSize.x
		|| oriWindow.y >= dc_renderParam.windowSize.y) return;

	glm::vec3 rayDrc;
	const glm::vec3& camPos = dc_renderParam.camPos2[blockIdx.z];
	const glm::mat4 unProjection = dc_renderParam.unProjection2[blockIdx.z];
	float tEnter, tExit;
	{
		// find Ray of each Pixel on Window
		//   unproject
		glm::vec4 v41 = unProjection * glm::vec4{
			(sbsmplTexVal.x - .5f) * 2.f,
			(sbsmplTexVal.y - .5f) * 2.f,
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
				oriWindow.x, oriWindow.y);
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
		(float)windowX / dc_renderParam.windowSize.x,
		(float)windowY / dc_renderParam.windowSize.y);
	d_color = rgbaFloatToUbyte4(reconsTexVal.x, reconsTexVal.y, reconsTexVal.z, 1.f);
}

__global__ void FAVRReconstruction(
	glm::u8vec4* d_colorL, glm::u8vec4* d_colorR)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.windowSize.x || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	glm::u8vec4& d_color = blockIdx.z == 0 ?
		d_colorL[windowFlatIdx] : d_colorR[windowFlatIdx];
	float scale = 1.f / dc_renderParam.sbsmplLvl;
	float4 reconsTexVal = tex2D<float4>(
		dc_reconsTexes[dc_renderParam.sbsmplLvl - 1],
		(float)windowX / dc_renderParam.windowSize.x,
		(float)windowY / dc_renderParam.windowSize.y);
	float4 sbsmplVal = tex2D<float4>(dc_sbsmplColorTex2[blockIdx.z],
		reconsTexVal.x * scale, reconsTexVal.y);
	d_color = rgbaFloatToUbyte4(sbsmplVal.x, sbsmplVal.y, sbsmplVal.z, sbsmplVal.w);
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::render(
	uint32_t windowW, uint32_t windowH, uint8_t sbsmplLvl)
{
	if (stream == nullptr)
		CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

#ifndef NO_FAVR
	assert(sbsmplLvl > 0 && sbsmplLvl <= MAX_SUBSAMPLE_LEVEL_NUM);
	if (d_sbsmplTexArrs[sbsmplLvl - 1] == nullptr)
		createSubsampleAndReconstructionTexes(sbsmplLvl, windowW, windowH);
#endif // !NO_FAVR

	// from here, called per frame, thus no CUDA_RUNTIME_API_CHECK
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

	dim3 threadPerBlock = { 16, 16 };
	dim3 blockPerGrid;
#define TEST_SUBSAMPLE_TEX
#ifdef NO_FAVR

	blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
	renderKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
		d_color2[0], d_color2[1], d_depthTex2[0], d_depthTex2[1]);

#elif defined(TEST_RECONSTRUCTION_TEX)

	blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
	testReconstructionTexKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
		d_color2[0], d_color2[1]);

#else

	blockPerGrid = { (windowW / sbsmplLvl + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };

#	ifdef TEST_SUBSAMPLE_TEX

	testSubsampleTexKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
		d_color2[0], d_color2[1]);

#	elif defined(TEST_SUBSAMPLE)

	FAVRSubsample << < blockPerGrid, threadPerBlock, 0, stream >> > (
		d_color2[0], d_color2[1], d_depthTex2[0], d_depthTex2[1]);

#	else

	FAVRSubsample << < blockPerGrid, threadPerBlock, 0, stream >> > (
		d_sbsmplColor2[0], d_sbsmplColor2[1], d_depthTex2[0], d_depthTex2[1]);
	cudaMemcpyToArray(d_sbsmplColorTexArr2[0], 0, 0, d_sbsmplColor2[0],
		d_colorSize, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(d_sbsmplColorTexArr2[1], 0, 0, d_sbsmplColor2[1],
		d_colorSize, cudaMemcpyDeviceToDevice);

	blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
	FAVRReconstruction << < blockPerGrid, threadPerBlock, 0, stream >> > (
		d_color2[0], d_color2[1]);

#	endif
#endif // NO_FAVR

	for (uint8_t idx = 0; idx < 2; ++idx)
	{
		cudaMemcpyToArray(d_colorArr2[idx], 0, 0,
			d_color2[idx], d_colorSize, cudaMemcpyDeviceToDevice);

		d_colorArr2[idx] = d_depthArr2[idx] = nullptr;
		cudaGraphicsUnmapResources(1, &outColorTexRsc2[idx], stream);
		cudaGraphicsUnmapResources(1, &inDepthTexRsc2[idx], stream);
	}
}
