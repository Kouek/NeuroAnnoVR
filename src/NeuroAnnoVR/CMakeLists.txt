set(TARGET_NAME "NeuroAnnoVR")

aux_source_directory("./" SRCS)
aux_source_directory("./GUI" GUI_SRCS)

add_executable(
	${TARGET_NAME}
	${SRCS} ${GUI_SRCS}
	"${THIRDPARTY_DIR}/glad/src/glad.c"
	# Necessary! Or it won't be MOC
	"./GUI/EditorWindow.ui"
	"./GUI/LeftHandUI.ui"
	"./GUI/RightHandUI.ui"
	"${INCLUDE_DIR}/util/QTransferFunction.h"
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