set(TARGET_NAME "NeuroAnnoVR")

aux_source_directory(${CMAKE_CURRENT_LIST_DIR} SRCS)

add_executable(
	${TARGET_NAME}
	${SRCS}
	"EditorWindow.ui" # Necessary! Or it won't be MOC
	"${THIRDPARTY_DIR}/glad/src/glad.c"
)
target_include_directories(
	${TARGET_NAME}
	PRIVATE
	${OPENVR_INC_DIR}
)
LIST(
	APPEND
	LIBS
	${Qt5_LIBS}
	"CompVolumeRenderer"
)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	# No need to link OpenVR shared libs again,
	# since they were copied to executable output path
	target_link_libraries(
		${TARGET_NAME}
		${OPENVR_LIBS}
		${LIBS}
	)
endif()