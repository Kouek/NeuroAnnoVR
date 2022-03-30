set(TARGET_NAME "NeuroAnnoVR")

add_executable(
	${TARGET_NAME}
	"main.cpp"
	"${THIRDPARTY_DIR}/glad/src/glad.c"
)
if(WIN32)
	message("WIN32: Linking ${OPENVR_BIN_DIR}/win64/openvr_api.dll")
	target_link_libraries(
		${TARGET_NAME}
		PRIVATE
		"${OPENVR_BIN_DIR}/win64/openvr_api.dll"
	)
endif()