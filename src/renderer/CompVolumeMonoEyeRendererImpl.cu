#include "CompVolumeMonoEyeRendererImpl.h"

#include <Common/cuda_utils.hpp>

#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

namespace kouek
{
	namespace CompVolumeMonoEyeRendererImplCUDA
	{
		__constant__ CUDAParameter d_cudaParam;
		void uploadCUDAParameter(const CUDAParameter* hostMemDat)
		{
			CUDA_RUNTIME_CHECK(
				cudaMemcpyToSymbol(d_cudaParam, hostMemDat, sizeof(CUDAParameter)));
		}

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
		__constant__ glm::uvec4* d_mappingTableStride4 = nullptr;
		void uploadMappingTable(const uint32_t* hostMemDat, size_t size)
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

		__device__ bool virtualSampleLOD0(
			float* sampleVal, const glm::vec3& samplePos)
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
					* (float)d_compVolumeParam.blockLength
					+ offsetInNoPaddingBlock + (float)d_compVolumeParam.padding;
				// normolized
				GPUMemSamplePos /= d_cudaParam.texUnitDim;
			}

			*sampleVal = tex3D<float>(d_textures[GPUMemBlockIdx.w & (0x0000ffff)],
				GPUMemSamplePos.x, GPUMemSamplePos.y, GPUMemSamplePos.z);
			return 1;
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

		__global__ void renderKernel(uint32_t* d_window)
		{
			uint32_t windowX = blockIdx.x * blockDim.x + threadIdx.x;
			uint32_t windowY = blockIdx.y * blockDim.y + threadIdx.y;
			if (windowX >= d_renderParam.windowSize.x || windowY >= d_renderParam.windowSize.y) return;
			size_t windowFlatIdx = (size_t)windowY * d_renderParam.windowSize.x + windowX;

			d_window[windowFlatIdx] = rgbaFloatToUInt32(
				d_renderParam.lightParam.bkgrndColor.r,
				d_renderParam.lightParam.bkgrndColor.g,
				d_renderParam.lightParam.bkgrndColor.b,
				d_renderParam.lightParam.bkgrndColor.a);

			glm::vec3 rayDrc;
			float tEnter, tExit;
			{
				// find Ray of each Pixel on Window
				//   unproject
				glm::vec4 v41 = d_renderParam.unProjection * glm::vec4{
					(((float)windowX / d_renderParam.windowSize.x) - .5f) * 2.f,
					(((float)windowY / d_renderParam.windowSize.y) - .5f) * 2.f,
					1.f, 1.f };
				//   don't rotate first to compute the near&far-clip steps
				rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;
				rayDrc = glm::normalize(rayDrc);
				float tNearClip = d_renderParam.nearClip / fabsf(rayDrc.z);
				float tFarClip = d_renderParam.farClip / fabsf(rayDrc.z);
				//   rotate
				v41 = d_renderParam.camRotaion * v41;
				rayDrc.x = v41.x, rayDrc.y = v41.y, rayDrc.z = v41.z;
				rayDrc = glm::normalize(rayDrc);

				// Ray intersect Subregion(OBB)
				// equivalent to Ray intersect AABB in Subreion Space
				//   for pos, apply Rotation and Translation
				glm::vec4 v42{ d_renderParam.camPos.x, d_renderParam.camPos.y,
					d_renderParam.camPos.z, 1.f };
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
				
				// near&far-clip
				if (tEnter < tNearClip) tEnter = tNearClip;
				if (tExit > tFarClip) tExit = tFarClip;
			}

#ifdef TEST_RAY_DIRECTION
			// TEST: Ray Direction
			d_window[windowFlatIdx] = rgbaFloatToUInt32(rayDrc.x, rayDrc.y, rayDrc.z, 1.f);
			return;
#endif // TEST_RAY_DIRECTION

			// no intersection
			if (tEnter >= tExit)
				return;
			glm::vec3 rayPos = d_renderParam.camPos + tEnter * rayDrc;


#ifdef TEST_RAY_ENTER_POSITION
			// TEST: Ray Enter Position
			d_window[windowFlatIdx] = rgbaFloatToUInt32(
				.5f * rayPos.x / d_renderParam.subrgn.halfW,
				.5f * rayPos.y / d_renderParam.subrgn.halfH,
				.5f * rayPos.z / d_renderParam.subrgn.halfD, 1.f);
			return;
#endif // TEST_RAY_ENTER_POSITION

			glm::vec3 subrgnCenterInWdSp = {
				.5f * d_renderParam.subrgn.halfW,
				.5f * d_renderParam.subrgn.halfH,
				.5f * d_renderParam.subrgn.halfD,
			};
			glm::vec3 samplePos;
			glm::vec4 color = glm::zero<glm::vec4>();
			float sampleVal = 0;
			uint32_t stepNum = 0;
			for (; stepNum <= d_renderParam.maxStepNum && tEnter <= d_renderParam.maxStepDist;
				++stepNum,
				tEnter += d_renderParam.step,
				rayPos += rayDrc * d_renderParam.step)
			{
				// ray pos in World Space -> sample pos in Voxel Space
				samplePos =
					(rayPos - subrgnCenterInWdSp + d_renderParam.subrgn.center)
					/ d_compVolumeParam.spaces;

				// virtual sample in Voxel Space, real sample in GPU Mem
				float currSampleVal;
				{
					int flag = virtualSampleLOD0(&currSampleVal, samplePos);
					if (flag == 0 || currSampleVal <= 0)
						continue;
				}

				float4 currColor = tex2D<float4>(d_preIntTransferFunc, sampleVal, currSampleVal);
				if (currColor.w <= 0)
					continue;

				sampleVal = currSampleVal;
				color = color + (1.f - color.w) * glm::vec4{ currColor.x,currColor.y,currColor.z,currColor.w }
					* glm::vec4{ currColor.w,currColor.w,currColor.w,1.f };
				
				if (color.w > 0.9f)
					break;
			}

			d_window[windowFlatIdx] = rgbaFloatToUInt32(color.r, color.g, color.b, color.a);
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
