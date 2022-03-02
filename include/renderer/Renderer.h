#ifndef KOUEK_RENDERER_H
#define KOUEK_RENDERER_H

#include <glad/glad.h>

#include <renderer/math.h>

#include <VolumeSlicer/volume.hpp>
#include <VolumeSlicer/transfer_function.hpp>

namespace kouek
{
	class CompVolumeRenderer
	{
	public:
		struct CUDAParameter
		{
			CUcontext ctx;
			uint32_t texUnitNum;
			std::array<uint32_t, 3> texUnitDim;
		};
		struct LightParamter
		{
			float ka;
			float kd;
			float ks;
			float shininess;
			float4 bkgrndColor;
		};
		/// <summary>
		/// Subregion is a zOx OBB bounding box in Ray-Casting space
		/// which restricts the volume block loading.
		/// In different renderers, the restriction may be different.
		/// </summary>
		struct Subregion
		{
			glm::vec3 center;
			glm::mat4 rotation;
			float halfW, halfH, halfD;
		};

		virtual void registerOutputGLPBO(GLuint outPBO, uint32_t w, uint32_t h) = 0;
		virtual void unregisterOutputGLPBO() = 0;
		virtual void setStep(uint32_t maxStepNum, float maxStepDist) = 0;
		virtual void setSubregion(const Subregion& subrgn) = 0;
		virtual void setTransferFunc(const vs::TransferFunc& tf) = 0;
		virtual void setLightParam(const LightParamter& lightParam) = 0;
		virtual void setVolume(std::shared_ptr<vs::CompVolume> volume) = 0;
		virtual void render() = 0;
	};

	class CompVolumeMonoEyeRenderer : public CompVolumeRenderer
	{
	public:
		static std::unique_ptr<CompVolumeMonoEyeRenderer> create(
			const CUDAParameter& cudaParam
		);

		virtual void registerOutputGLPBO(GLuint outPBO, uint32_t w, uint32_t h) = 0;
		virtual void unregisterOutputGLPBO() = 0;
		virtual void setStep(uint32_t maxStepNum, float maxStepDist) = 0;
		virtual void setSubregion(const Subregion& subrgn) = 0;
		virtual void setTransferFunc(const vs::TransferFunc& tf) = 0;
		virtual void setLightParam(const LightParamter& lightParam) = 0;
		virtual void setVolume(std::shared_ptr<vs::CompVolume> volume) = 0;
		virtual void render() = 0;

		virtual void setCamera(
			const glm::vec3& pos,
			const glm::mat4& rotation,
			const glm::mat4& unProjection) = 0;
	};

	class CompVolumeDualEyeRenderer : public CompVolumeRenderer
	{

	};
}

#endif // !KOUEK_RENDERER_H
