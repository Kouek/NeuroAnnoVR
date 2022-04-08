#include "CompVolumeMonoEyeRendererImpl.h"

#include <Common/cuda_utils.hpp>

#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

using namespace kouek::CompVolumeRendererCUDA;

// CUDA Resource:
//   Allocated when needed,
//   freeed when CompVolumeRendererCUDA::MonoEyeFunc is deconstructed
__constant__ CompVolumeParameter dc_compVolumeParam;
__constant__ MonoEyeRenderParameter dc_renderParam;

__constant__ uint32_t dc_blockOffsets[MAX_LOD + 1];
__constant__ cudaTextureObject_t dc_textures[MAX_TEX_UNIT_NUM];

__constant__ cudaTextureObject_t dc_transferFunc;

cudaArray_t d_preIntTFArray = nullptr;
cudaTextureObject_t d_preIntTF;
__constant__ cudaTextureObject_t dc_preIntTransferFunc;

uint32_t* d_mappingTable = nullptr;
__constant__ glm::uvec4* d_mappingTableStride4 = nullptr;

cudaGraphicsResource_t outColorTexRsc = nullptr, inDepthTexRsc = nullptr;
glm::u8vec4* d_color = nullptr;
struct
{
	cudaResourceDesc rscDesc;
	cudaTextureDesc texDesc;
	cudaTextureObject_t tex;
}d_depth;
size_t d_colorSize;
cudaArray_t d_colorArr = nullptr, d_depthArr = nullptr;
cudaStream_t stream = nullptr;

kouek::CompVolumeRendererCUDA::MonoEyeFunc::~MonoEyeFunc()
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

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::uploadCompVolumeParam(const CompVolumeParameter& param)
{
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_compVolumeParam, &param, sizeof(CompVolumeParameter)));
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::uploadRenderParam(const MonoEyeRenderParameter& param)
{
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_renderParam, &param, sizeof(MonoEyeRenderParameter)));
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::uploadBlockOffs(const uint32_t* hostMemDat, size_t num)
{
	assert(num <= MAX_LOD + 1);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_blockOffsets, hostMemDat, sizeof(uint32_t) * num));
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::uploadCUDATextureObj(const cudaTextureObject_t* hostMemDat, size_t num)
{
	assert(num <= MAX_TEX_UNIT_NUM);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_textures, hostMemDat, sizeof(cudaTextureObject_t) * num));
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::uploadTransferFunc(const float* hostMemDat)
{
	// TODO
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::uploadPreIntTransferFunc(const float* hostMemDat)
{
	if (d_preIntTFArray == nullptr)
		CreateCUDATexture2D(256, 256, &d_preIntTFArray, &d_preIntTF);
	UpdateCUDATexture2D(
		(uint8_t*)hostMemDat, d_preIntTFArray, sizeof(float) * 256 * 4, 256, 0, 0);
	CUDA_RUNTIME_CHECK(
		cudaMemcpyToSymbol(dc_preIntTransferFunc, &d_preIntTF, sizeof(cudaTextureObject_t)));
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::uploadMappingTable(const uint32_t* hostMemDat, size_t size)
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

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::registerGLResource(GLuint outColorTex, GLuint inDepthTex, uint32_t w, uint32_t h)
{
	d_colorSize = sizeof(glm::u8vec4) * w * h;
	CUDA_RUNTIME_API_CALL(
		cudaGraphicsGLRegisterImage(&outColorTexRsc, outColorTex,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	CUDA_RUNTIME_API_CALL(
		cudaMalloc(&d_color, d_colorSize));

	CUDA_RUNTIME_API_CALL(
		cudaGraphicsGLRegisterImage(&inDepthTexRsc, inDepthTex,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	memset(&d_depth.rscDesc, 0, sizeof(cudaResourceDesc));
	d_depth.rscDesc.resType = cudaResourceTypeArray;
	memset(&d_depth.texDesc, 0, sizeof(cudaTextureDesc));
	d_depth.texDesc.normalizedCoords = 0;
	d_depth.texDesc.filterMode = cudaFilterModePoint;
	d_depth.texDesc.addressMode[0] = cudaAddressModeClamp;
	d_depth.texDesc.addressMode[1] = cudaAddressModeClamp;
	d_depth.texDesc.readMode = cudaReadModeElementType;
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::unregisterGLResource()
{
	if (outColorTexRsc != nullptr)
	{
		CUDA_RUNTIME_API_CALL(cudaGraphicsUnregisterResource(outColorTexRsc));
		outColorTexRsc = nullptr;
		CUDA_RUNTIME_API_CALL(cudaFree(d_color));
		d_color = nullptr;

		CUDA_RUNTIME_API_CALL(cudaGraphicsUnregisterResource(inDepthTexRsc));
		inDepthTexRsc = nullptr;
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
		* powf(fmaxf(dot(N, (L + R) / 2.f), 0),
			dc_renderParam.lightParam.shininess));
	glm::vec3 diffuse = dc_renderParam.lightParam.kd
		* fmaxf(dot(N, L), 0.f) * diffuseColor;

	return ambient + specular + diffuse;
}

// WARNING:
// - Declaring type of param d_depth as [const cudaTextureObject_t &]
//   will cause unknown error at tex2D()
__global__ void renderKernel(glm::u8vec4* d_color, cudaTextureObject_t d_depthTex)
{
	uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
	if (windowX >= dc_renderParam.windowSize.x || windowY >= dc_renderParam.windowSize.y) return;
	size_t windowFlatIdx = (size_t)windowY * dc_renderParam.windowSize.x + windowX;

	d_color[windowFlatIdx] = rgbaFloatToUbyte4(
		dc_renderParam.lightParam.bkgrndColor.r,
		dc_renderParam.lightParam.bkgrndColor.g,
		dc_renderParam.lightParam.bkgrndColor.b,
		dc_renderParam.lightParam.bkgrndColor.a);

	glm::vec3 rayDrc;
	float tEnter, tExit;
	{
		// find Ray of each Pixel on Window
		//   unproject
		glm::vec4 v41 = dc_renderParam.unProjection * glm::vec4{
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
			uchar4 depth4 = tex2D<uchar4>(d_depthTex, windowX, windowY);
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
		glm::vec4 v42{ dc_renderParam.camPos.x, dc_renderParam.camPos.y,
			dc_renderParam.camPos.z, 1.f };
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

#ifdef TEST_RAY_DIRECTION
	// TEST: Ray Direction
	d_color[windowFlatIdx] = rgbaFloatToUbyte4(rayDrc.x, rayDrc.y, rayDrc.z, 1.f);
	return;
#endif // TEST_RAY_DIRECTION

	// no intersection
	if (tEnter >= tExit)
		return;
	glm::vec3 rayPos = dc_renderParam.camPos + tEnter * rayDrc;

#ifdef TEST_RAY_ENTER_EXIT_DIFF
	// TEST: Ray Enter Difference
	float diff = tExit - tEnter;
	d_color[windowFlatIdx] = rgbaFloatToUbyte4(diff, diff, diff, 1.f);
	return;
#endif // TEST_RAY_ENTER_EXIT_DIFF

#ifdef TEST_RAY_ENTER_POSITION
	// TEST: Ray Enter Position
	d_color[windowFlatIdx] = rgbaFloatToUbyte4(
		.5f * rayPos.x / d_renderParam.subrgn.halfW,
		.5f * rayPos.y / d_renderParam.subrgn.halfH,
		.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
	return;
#endif // TEST_RAY_ENTER_POSITION

#ifdef TEST_RAY_EXIT_POSITION
	// TEST: Ray Exit Position
	rayPos = d_renderParam.camPos + tExit * rayDrc;
	d_color[windowFlatIdx] = rgbaFloatToUbyte4(
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

	d_color[windowFlatIdx] = rgbaFloatToUbyte4(color.r, color.g, color.b, color.a);
}

void kouek::CompVolumeRendererCUDA::MonoEyeFunc::render(uint32_t windowW, uint32_t windowH)
{
	if (stream == nullptr)
		CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

	// from here, called per frame, thus no CUDA_RUNTIME_API_CHECK
	cudaGraphicsMapResources(1, &outColorTexRsc, stream);
	cudaGraphicsSubResourceGetMappedArray(&d_colorArr, outColorTexRsc, 0, 0);

	cudaGraphicsMapResources(1, &inDepthTexRsc, stream);
	cudaGraphicsSubResourceGetMappedArray(&d_depthArr, inDepthTexRsc, 0, 0);
	d_depth.rscDesc.res.array.array = d_depthArr;
	cudaCreateTextureObject(&d_depth.tex, &d_depth.rscDesc,
		&d_depth.texDesc, nullptr);

	dim3 threadPerBlock = { 16, 16 };
	dim3 blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
						 (windowH + threadPerBlock.y - 1) / threadPerBlock.y };
	renderKernel <<< blockPerGrid, threadPerBlock, 0, stream >>> (d_color, d_depth.tex);

	cudaMemcpyToArray(d_colorArr, 0, 0,
		d_color, d_colorSize, cudaMemcpyDeviceToDevice);

	d_colorArr = d_depthArr = nullptr;
	cudaGraphicsUnmapResources(1, &outColorTexRsc, stream);
	cudaGraphicsUnmapResources(1, &inDepthTexRsc, stream);
}
