#ifndef KOUEK_RENDERER_H
#define KOUEK_RENDERER_H

#include <glad/glad.h>

#include <util/Math.h>

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
			glm::uvec3 texUnitDim;
		};
		struct LightParamter
		{
			float ka;
			float kd;
			float ks;
			float shininess;
			glm::vec4 bkgrndColor;
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
			glm::mat4 fromWorldToSubrgn;
			float halfW, halfH, halfD;
		};

		virtual void setStep(uint32_t maxStepNum, float step) = 0;
		virtual void setSubregion(const Subregion& subrgn) = 0;
		virtual void setTransferFunc(const vs::TransferFunc& tf) = 0;
		virtual void setLightParam(const LightParamter& lightParam) = 0;
		virtual void setVolume(std::shared_ptr<vs::CompVolume> volume) = 0;
		virtual void setSpacesScale(float scale) = 0;
		virtual void render() = 0;
	};

	class CompVolumeMonoEyeRenderer : virtual public CompVolumeRenderer
	{
	public:
		static std::unique_ptr<CompVolumeMonoEyeRenderer> create(
			const CUDAParameter& cudaParam);

		virtual void registerGLResource(
			GLuint outColorTex, GLuint inDepthTex,
			uint32_t w, uint32_t h) = 0;
		virtual void unregisterGLResource() = 0;

		struct CameraParameter
		{
			const glm::vec3& pos;
			const glm::mat4& rotation;
			const glm::mat4& unProjection;
			float nearClip = .01f, farClip = 100.f;
		};
		virtual void setCamera(const CameraParameter& camParam) = 0;
	};

	class CompVolumeFAVRRenderer : virtual public CompVolumeRenderer
	{
	public:
		static std::unique_ptr<CompVolumeFAVRRenderer> create(
			const CUDAParameter& cudaParam);

		virtual void registerGLResource(
			GLuint outLftColorTex, GLuint outRhtColorTex,
			GLuint inLftDepthTex, GLuint inRhtDepthTex,
			uint32_t w, uint32_t h) = 0;
		virtual void unregisterGLResource() = 0;

		enum class RenderTarget : uint8_t
		{
			Image = 0,
			SubsampleTex,
			SubsampleResult,
			ReconstructionTex,
			ReconstructionXDiff,
			ReconstructionYDiff,
			FullResolutionImage,
			Last
		};
		struct CameraParameter
		{
			const glm::vec3& lftEyePos, rhtEyePos;
			const glm::mat4& rotation;
			const glm::mat4& lftUnProjection, rhtUnProjection;
			float nearClip = .01f, farClip = 100.f;
		};
		enum class InteractionMode : uint8_t
		{
			AnnotationBall = 0,
			AnnotationLaser
		};
		virtual void setCamera(const CameraParameter& camParam) = 0;
		struct InteractionParameter
		{
			InteractionMode mode;
			union
			{
				struct
				{
					glm::vec3 AABBSize;
					glm::vec3 startPos;
				}ball;
				struct
				{
					glm::vec3 ori;
					glm::vec3 drc;
				}laser;
			}dat;
		};
		virtual void setInteractionParam(const InteractionParameter& intrctParam) = 0;
		virtual void render() = 0;
		virtual void render(glm::vec3* intrctPos, RenderTarget renderTar) = 0;
	};
}

#endif // !KOUEK_RENDERER_H
