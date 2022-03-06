set(TARGET_NAME "CompVolumeRenderer")

foreach(
	vscore_src
	"Volume/volume_impl.cpp"
    "Volume/block_loader.cpp"
    "Volume/volume_cache_impl.cpp"
	"IO/reader_impl.cpp"
    "Common/cuda_context.cpp"
	"Common/image.cpp"
    "Common/library.cpp"
    "Common/library_reposity.cpp"
	"Common/plugin_loader.cpp"
)
	list(APPEND VSCORE_SRCS "${CMAKE_CURRENT_LIST_DIR}/VSCore/${vscore_src}")
endforeach()
message(STATUS "${TARGET_NAME}'s VSCORE_SRCS: ${VSCORE_SRCS}")

aux_source_directory(${CMAKE_CURRENT_LIST_DIR} SRCS)
message(STATUS "${TARGET_NAME}'s SRCS: ${SRCS}")

add_library(
	${TARGET_NAME}
	STATIC
	${VSCORE_SRCS}
	${SRCS}
)
target_include_directories(
	${TARGET_NAME}
	PUBLIC
	${INCLUDE_DIR}
	"${INCLUDE_DIR}/renderer/VSCore" # VSCore header-only source file
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	PRIVATE
	"${CMAKE_CURRENT_LIST_DIR}/VSCore" # VSCore inner header
)
target_link_libraries(
	${TARGET_NAME}
	PUBLIC
	"spdlog::spdlog"
	"glm::glm"
	PRIVATE
	"voxel_uncompress"
)
