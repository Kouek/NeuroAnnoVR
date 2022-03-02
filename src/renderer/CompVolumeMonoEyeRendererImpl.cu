#include "CompVolumeMonoEyeRendererImpl.h"

#include <Common/cuda_utils.hpp>

#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

namespace kouek
{
	namespace CompVolumeMonoEyeRendererImplCUDA
	{
		__constant__ uint32_t d_blockOffsets[MAX_LOD + 1];
		void uploadBlockOffs(
			const uint32_t* hostMemDat, size_t num)
		{
			assert(num <= MAX_LOD + 1);
			CUDA_RUNTIME_CHECK(
				cudaMemcpyToSymbol(d_blockOffsets, hostMemDat, sizeof(uint32_t) * num));
		}

		__constant__ CompVolumeParameter d_compVolumeParam;
		void uploadCompVolumeParam(const CompVolumeParameter* hostMemDat)
		{
			CUDA_RUNTIME_CHECK(
				cudaMemcpyToSymbol(d_compVolumeParam, hostMemDat, sizeof(CompVolumeParameter)));
		}

		__constant__ cudaTextureObject_t d_textures[MAX_TEX_UNIT_NUM];
		void uploadCUDATextureObj(
			const cudaTextureObject_t* hostMemDat, size_t num)
		{
			assert(num <= MAX_TEX_UNIT_NUM);
			CUDA_RUNTIME_CHECK(
				cudaMemcpyToSymbol(d_textures, hostMemDat, sizeof(cudaTextureObject_t) * num));
		}

		__constant__ cudaTextureObject_t d_transferFunc;
		void uploadTransferFunc(
			const float* hostMemDat)
		{
			// TODO
		}

		cudaArray* preIntTFArray = nullptr;
		cudaTextureObject_t preIntTF;
		__constant__ cudaTextureObject_t d_preIntTransferFunc;
		void uploadPreIntTransferFunc(
			const float* hostMemDat)
		{
			if (preIntTFArray == nullptr)
				CreateCUDATexture2D(256, 256, &preIntTFArray, &preIntTF);
			UpdateCUDATexture2D(
				(uint8_t*)hostMemDat, preIntTFArray, sizeof(float) * 256 * 4, 256, 0, 0);
			CUDA_RUNTIME_CHECK(
				cudaMemcpyToSymbol(d_preIntTransferFunc, &preIntTF, sizeof(cudaTextureObject_t)));
		}

		__constant__ RenderParameter d_renderParam;
		void uploadRenderParam(const RenderParameter* hostMemDat)
		{
			CUDA_RUNTIME_CHECK(
				cudaMemcpyToSymbol(d_renderParam, hostMemDat, sizeof(RenderParameter)));
		}

		uint32_t* d_mappingTable = nullptr;
		__constant__ uint4* d_mappingTableStride4 = nullptr;
		void uploadMappingTable(const uint32_t* hostMemDat, size_t size)
		{
			if (d_mappingTable == nullptr)
			{
				cudaMalloc(&d_mappingTable, size);
				// cpy uint32_t ptr to uint4 ptr
				CUDA_RUNTIME_API_CALL(
					cudaMemcpyToSymbol(d_mappingTableStride4, &d_mappingTable, sizeof(uint4*)));
			}
			CUDA_RUNTIME_API_CALL(
				cudaMemcpy(d_mappingTable, hostMemDat, size, cudaMemcpyHostToDevice));
		}

		cudaGraphicsResource_t PBORsc = nullptr;
		void registerOutputGLPBO(GLuint outPBO)
		{
			CUDA_RUNTIME_API_CALL(
				cudaGraphicsGLRegisterBuffer(&PBORsc, outPBO, cudaGraphicsMapFlagsWriteDiscard));
		}

		void unregisterOutputGLPBO()
		{
			if (PBORsc != nullptr)
			{
				CUDA_RUNTIME_API_CALL(
					cudaGraphicsUnregisterResource(PBORsc));
				PBORsc = nullptr;
			}
		}

		__device__ uint32_t rgbaFloatToUInt32(float r, float g, float b, float a)
		{
			r = __saturatef(r); // clamp to [0.0, 1.0]
			g = __saturatef(g);
			b = __saturatef(b);
			a = __saturatef(a);
			return (uint32_t(r * 255) << 24) | (uint32_t(g * 255) << 16)
				| (uint32_t(b * 255) << 8) | uint32_t(a * 255);
		}

		__global__ void renderKernel(uint32_t* d_window)
		{
			uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
			uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
			if (windowX >= d_renderParam.windowSize.x || windowY >= d_renderParam.windowSize.y) return;
			size_t windowFlatIdx = (size_t)windowY * d_renderParam.windowSize.x + windowX;

			glm::vec3 rayDrc;
			{
				float offsX = (((float)windowX / d_renderParam.windowSize.x) - .5f) * 2.f;
				float offsY = (((float)windowY / d_renderParam.windowSize.y) - .5f) * 2.f;
				glm::vec4 v4 = d_renderParam.unProjection * glm::vec4(offsX, offsY, 0, 0);
				rayDrc = d_renderParam.rotaion * v4;
				rayDrc = glm::normalize(rayDrc);
			}
			d_window[windowFlatIdx] = rgbaFloatToUInt32(rayDrc.x, rayDrc.y, 1.f, 1.f);

			// intersect Subregion(OBB)
		}

		uint32_t* d_window = nullptr;
		cudaStream_t stream = nullptr;
		void render(uint32_t windowW, uint32_t windowH)
		{
			if (stream == nullptr)
				CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

			// from here, called per frame, thus no CUDA_RUNTIME_API_CHECK
			cudaGraphicsMapResources(1, &PBORsc, stream);
			size_t rscSize;
			cudaGraphicsResourceGetMappedPointer((void**)&d_window, &rscSize, PBORsc);

			dim3 threadPerBlock = { 16, 16 };
			dim3 blockPerGrid = { (windowW + threadPerBlock.x - 1) / threadPerBlock.x,
								 (windowH + threadPerBlock.y - 1) / threadPerBlock.y };
			renderKernel<<<blockPerGrid, threadPerBlock, 0, stream >>>(d_window);

			cudaGraphicsUnmapResources(1, &PBORsc, stream);
			d_window = nullptr;
		}
	}
}
