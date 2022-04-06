#include "CompVolumeFAVRRendererImpl.h"

#include <Common/cuda_utils.hpp>

#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

using namespace kouek::CompVolumeRendererCUDA;

// CUDA Resource:
//   Allocated when needed,
//   freeed when CompVolumeRendererCUDA::FAVRFunc is deconstructed
__constant__ CompVolumeParameter d_compVolumeParam;
__constant__ FAVRRenderParameter d_renderParam;

__constant__ uint32_t d_blockOffsets[MAX_LOD + 1];
__constant__ cudaTextureObject_t d_textures[MAX_TEX_UNIT_NUM];

__constant__ cudaTextureObject_t d_transferFunc;

cudaArray_t preIntTFArray = nullptr;
cudaTextureObject_t preIntTF;
__constant__ cudaTextureObject_t d_preIntTransferFunc;

uint32_t* d_mappingTable = nullptr;
__constant__ glm::uvec4* d_mappingTableStride4 = nullptr;

cudaArray_t sbsmplSurfArrs[MAX_SUBSAMPLE_LEVEL_NUM] = { nullptr };
cudaArray_t reconsSurfArrs[MAX_SUBSAMPLE_LEVEL_NUM] = { nullptr };
cudaSurfaceObject_t sbsmplSurfs[MAX_SUBSAMPLE_LEVEL_NUM];
cudaSurfaceObject_t reconsSurfs[MAX_SUBSAMPLE_LEVEL_NUM];
__constant__ float sbsmplRads[MAX_SUBSAMPLE_LEVEL_NUM - 1];
__constant__ cudaSurfaceObject_t d_sbsmplSurfs[MAX_SUBSAMPLE_LEVEL_NUM];
__constant__ cudaSurfaceObject_t d_reconsSurfs[MAX_SUBSAMPLE_LEVEL_NUM];

cudaGraphicsResource_t outColorTexRsc2[2] = { nullptr };
cudaGraphicsResource_t inDepthTexRsc2[2] = { nullptr };
glm::u8vec4* d_color2[2] = { nullptr };
struct
{
	cudaResourceDesc rscDesc;
	cudaTextureDesc texDesc;
}depthTexDesc;
cudaTextureObject_t d_depthTex2[2];
size_t d_colorSize;
cudaArray_t d_colorArr2[2] = { nullptr }, d_depthArr2[2] = { nullptr };
cudaStream_t stream = nullptr;

kouek::CompVolumeRendererCUDA::FAVRFunc::~FAVRFunc()
{
	if (preIntTFArray != nullptr)
	{
		CUDA_RUNTIME_CHECK(
			cudaDestroyTextureObject(preIntTF));
		CUDA_RUNTIME_CHECK(
			cudaFreeArray(preIntTFArray));
		preIntTFArray = nullptr;
	}
	// TODO
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadCompVolumeParam(const CompVolumeParameter& param)
{
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(d_compVolumeParam, &param, sizeof(CompVolumeParameter)));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadRenderParam(
	const FAVRRenderParameter& param)
{
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(d_renderParam, &param, sizeof(FAVRRenderParameter)));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadBlockOffs(const uint32_t* hostMemDat, size_t num)
{
	assert(num <= MAX_LOD + 1);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(d_blockOffsets, hostMemDat, sizeof(uint32_t) * num));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num)
{
	assert(num <= MAX_TEX_UNIT_NUM);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(d_textures, hostMemDat, sizeof(cudaTextureObject_t) * num));
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadTransferFunc(const float* hostMemDat)
{
	// TODO
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::uploadPreIntTransferFunc(const float* hostMemDat)
{
	if (preIntTFArray == nullptr)
		CreateCUDATexture2D(256, 256, &preIntTFArray, &preIntTF);
	UpdateCUDATexture2D(
		(uint8_t*)hostMemDat, preIntTFArray, sizeof(float) * 256 * 4, 256, 0, 0);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(d_preIntTransferFunc, &preIntTF, sizeof(cudaTextureObject_t)));
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

			CUDA_RUNTIME_API_CALL(cudaGraphicsUnregisterResource(inDepthTexRsc2[idx]));
			inDepthTexRsc2[idx] = nullptr;
		}
}

__global__ void createSubsampleSurfKernel(
	cudaSurfaceObject_t d_surf, uint8_t lvl)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX > d_renderParam.windowSize.x || windowY > d_renderParam.windowSize.y) return;
	glm::vec2 XYToCenter = {
		(((float)windowX / d_renderParam.windowSize.x) - .5f) * 2.f,
		(((float)windowY / d_renderParam.windowSize.y) - .5f) * 2.f };
	for (uint8_t i = 2; i <= lvl; ++i)
	{

	}
}

//static void createSubsampleAndReconsSurf(uint8_t lvl, uint32_t w, uint32_t h)
//{
//	assert((float)w / lvl >= h);
//	for (uint8_t i = 2; i <= lvl; ++i)
//	{
//		float rad = (float)w / i / 2.f;
//		CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(
//			sbsmplRads[i - 2], &rad, sizeof(float)));
//	}
//
//	uint8_t idx = lvl - 1;
//	cudaChannelFormatDesc chnnlDesc =
//		cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindUnsigned);
//	CUDA_RUNTIME_API_CALL(
//		cudaMallocArray(&sbsmplSurfArrs[idx], &chnnlDesc,
//			w, h, cudaArraySurfaceLoadStore));
//	cudaResourceDesc rscDesc;
//	memset(&rscDesc, 0, sizeof(cudaResourceDesc));
//	rscDesc.resType = cudaResourceTypeArray;
//	rscDesc.res.array.array = sbsmplSurfArrs[idx];
//	CUDA_RUNTIME_API_CALL(cudaCreateSurfaceObject(&sbsmplSurfs[idx], &rscDesc));
//	CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(
//		d_sbsmplSurfs[idx], &sbsmplSurfs[idx], sizeof(cudaSurfaceObject_t));
//
//	dim3 threadPerBlock = { 16, 16 };
//	dim3 blockPerGrid = { (w + threadPerBlock.x - 1) / threadPerBlock.x,
//						 (h + threadPerBlock.y - 1) / threadPerBlock.y };
//	createSubsampleSurfKernel << < blockPerGrid, threadPerBlock, 0, stream >> > (
//		sbsmplSurfs[idx], lvl, w, h);
//}

__device__ float virtualSampleLOD0(const glm::vec3& samplePos)
{
	// sample pos in Voxel Space -> virtual sample Block idx
	glm::uvec3 vsBlockIdx =
		samplePos / (float)d_compVolumeParam.noPaddingBlockLength;

	// virtual sample Block idx -> real sample Block idx (in GPU Mem)
	glm::uvec4 GPUMemBlockIdx;
	{
		size_t flatVSBlockIdx = d_blockOffsets[0]
			+ vsBlockIdx.z * d_compVolumeParam.LOD0BlockDim.y * d_compVolumeParam.LOD0BlockDim.x
			+ vsBlockIdx.y * d_compVolumeParam.LOD0BlockDim.x
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
			glm::vec3{ vsBlockIdx * d_compVolumeParam.noPaddingBlockLength };
		GPUMemSamplePos = glm::vec3{ GPUMemBlockIdx.x, GPUMemBlockIdx.y, GPUMemBlockIdx.z }
			*(float)d_compVolumeParam.blockLength
			+ offsetInNoPaddingBlock + (float)d_compVolumeParam.padding;
		// normolized
		GPUMemSamplePos /= d_renderParam.texUnitDim;
	}

	return tex3D<float>(d_textures[GPUMemBlockIdx.w & (0x0000ffff)],
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

	glm::vec3 ambient = d_renderParam.lightParam.ka * diffuseColor;
	glm::vec3 specular = glm::vec3(d_renderParam.lightParam.ks
		* powf(fmaxf(dot(N, (L + R) / 2.f), 0),
			d_renderParam.lightParam.shininess));
	glm::vec3 diffuse = d_renderParam.lightParam.kd
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
	if (windowX >= d_renderParam.windowSize.x || windowY >= d_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * d_renderParam.windowSize.x + windowX;

	if (blockIdx.z == 0) // render Left Eye 
		d_colorL[windowFlatIdx] = rgbaFloatToUbyte4(
			d_renderParam.lightParam.bkgrndColor.r,
			d_renderParam.lightParam.bkgrndColor.g,
			d_renderParam.lightParam.bkgrndColor.b,
			d_renderParam.lightParam.bkgrndColor.a);
	else // render Right Eye
		d_colorR[windowFlatIdx] = rgbaFloatToUbyte4(
			d_renderParam.lightParam.bkgrndColor.r,
			d_renderParam.lightParam.bkgrndColor.g,
			d_renderParam.lightParam.bkgrndColor.b,
			d_renderParam.lightParam.bkgrndColor.a);

	glm::vec3 rayDrc;
	const glm::vec3& camPos = d_renderParam.camPos2[blockIdx.z];
	const glm::mat4 unProjection = d_renderParam.unProjection2[blockIdx.z];
	float tEnter, tExit;
	{
		// find Ray of each Pixel on Window
		//   unproject
		glm::vec4 v41 = unProjection * glm::vec4{
			(((float)windowX / d_renderParam.windowSize.x) - .5f) * 2.f,
			(((float)windowY / d_renderParam.windowSize.y) - .5f) * 2.f,
			1.f, 1.f };
		//   don't rotate first to compute the Near&Far-clip steps
		rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;
		rayDrc = glm::normalize(rayDrc);
		float absRayDrcZ = fabsf(rayDrc.z);
		float tNearClip = d_renderParam.nearClip / absRayDrcZ;
		float tFarClip = d_renderParam.farClip;
		//   then compute upper bound of steps
		//   for Mesh-Volume mixed rendering
		{
			uchar4 depth4 = blockIdx.z == 0 ?
				tex2D<uchar4>(d_depthTexL, windowX, windowY) :
				tex2D<uchar4>(d_depthTexR, windowX, windowY);
			float meshBoundDep = d_renderParam.projection23 /
				((depth4.x / 255.f * 2.f - 1.f) + d_renderParam.projection22);
			if (tFarClip > meshBoundDep)
				tFarClip = meshBoundDep;
		} 
		tFarClip /= absRayDrcZ;
		//   rotate
		v41.x = rayDrc.x, v41.y = rayDrc.y, v41.z = rayDrc.z; // normalized in vec3
		v41 = d_renderParam.camRotaion * v41;
		rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;

		// Ray intersect Subregion(OBB)
		// equivalent to Ray intersect AABB in Subreion Space
		//   for pos, apply Rotation and Translation
		glm::vec4 v42{ camPos.x, camPos.y, camPos.z, 1.f };
		v42 = d_renderParam.subrgn.fromWorldToSubrgn * v42;
		//   for drc, apply Rotation only
		v41.w = 0;
		v41 = d_renderParam.subrgn.fromWorldToSubrgn * v41;
		rayIntersectAABB(
			&tEnter, &tExit,
			glm::vec3(v42),
			glm::normalize(glm::vec3(v41)),
			glm::zero<glm::vec3>(),
			glm::vec3{
				d_renderParam.subrgn.halfW * 2,
				d_renderParam.subrgn.halfH * 2,
				d_renderParam.subrgn.halfD * 2 });

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
	if (blockIdx.z == 0)
		d_colorL[windowFlatIdx] = rgbaFloatToUbyte4(
			.5f * rayPos.x / d_renderParam.subrgn.halfW,
			.5f * rayPos.y / d_renderParam.subrgn.halfH,
			.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
	else
		d_colorR[windowFlatIdx] = rgbaFloatToUbyte4(
			.5f * rayPos.x / d_renderParam.subrgn.halfW,
			.5f * rayPos.y / d_renderParam.subrgn.halfH,
			.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
	return;
#endif // TEST_RAY_ENTER_POSITION

#ifdef TEST_RAY_EXIT_POSITION
	// TEST: Ray Exit Position
	rayPos = camPos + tExit * rayDrc;
	if (blockIdx.z == 0)
		d_colorL[windowFlatIdx] = rgbaFloatToUbyte4(
			.5f * rayPos.x / d_renderParam.subrgn.halfW,
			.5f * rayPos.y / d_renderParam.subrgn.halfH,
			.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
	else
		d_colorR[windowFlatIdx] = rgbaFloatToUbyte4(
			.5f * rayPos.x / d_renderParam.subrgn.halfW,
			.5f * rayPos.y / d_renderParam.subrgn.halfH,
			.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
	return;
#endif // TEST_RAY_EXIT_POSITION

	glm::vec3 subrgnCenterInWdSp = {
		.5f * d_renderParam.subrgn.halfW,
		.5f * d_renderParam.subrgn.halfH,
		.5f * d_renderParam.subrgn.halfD,
	};
	glm::vec3 rayDrcMulStp = rayDrc * d_renderParam.step;
	glm::vec3 samplePos;
	glm::vec4 color = glm::zero<glm::vec4>();
	float sampleVal = 0;
	uint32_t stepNum = 0;
	for (;
		stepNum <= d_renderParam.maxStepNum && tEnter <= tExit;
		++stepNum, tEnter += d_renderParam.step, rayPos += rayDrcMulStp)
	{
		// ray pos in World Space -> sample pos in Voxel Space
		samplePos =
			(rayPos - subrgnCenterInWdSp + d_renderParam.subrgn.center)
			/ d_compVolumeParam.spaces;

		// virtual sample in Voxel Space, real sample in GPU Mem
		float currSampleVal = virtualSampleLOD0(samplePos);
		if (currSampleVal <= 0)
			continue;

		float4 currColor = tex2D<float4>(d_preIntTransferFunc, sampleVal, currSampleVal);
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

	if (blockIdx.z == 0)
		d_colorL[windowFlatIdx] = rgbaFloatToUbyte4(color.r, color.g, color.b, color.a);
	else
		d_colorR[windowFlatIdx] = rgbaFloatToUbyte4(color.r, color.g, color.b, color.a);
}

void kouek::CompVolumeRendererCUDA::FAVRFunc::render(
	uint32_t windowW, uint32_t windowH, uint8_t sbsmplLvl)
{
	if (stream == nullptr)
		CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

	assert(sbsmplLvl > 0 && sbsmplLvl <= MAX_SUBSAMPLE_LEVEL_NUM);
	//if (sbsmplSurfArrs[sbsmplLvl - 1] == nullptr)
	//	createSubsampleAndReconsSurf(sbsmplLvl, windowW, windowH);

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
	dim3 blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y, 2 };
	renderKernel <<< blockPerGrid, threadPerBlock, 0, stream >>> (
		d_color2[0], d_color2[1], d_depthTex2[0], d_depthTex2[1]);

	for (uint8_t idx = 0; idx < 2; ++idx)
	{
		cudaMemcpyToArray(d_colorArr2[idx], 0, 0,
			d_color2[idx], d_colorSize, cudaMemcpyDeviceToDevice);

		d_colorArr2[idx] = d_depthArr2[idx] = nullptr;
		cudaGraphicsUnmapResources(1, &outColorTexRsc2[idx], stream);
		cudaGraphicsUnmapResources(1, &inDepthTexRsc2[idx], stream);
	}
}
