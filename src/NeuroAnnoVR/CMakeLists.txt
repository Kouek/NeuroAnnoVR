set(TARGET_NAME "NeuroAnnoVR")

add_executable(
	${TARGET_NAME}
	"main.cpp"
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