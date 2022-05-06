set(TARGET_NAME "testQTransferFunction")

add_executable(
	${TARGET_NAME}
	"main.cpp"
	"${INCLUDE_DIR}/util/QTransferFunction.h" # Necessary! Or it won't be MOC
)
target_include_directories(
	${TARGET_NAME}
	PRIVATE
	${INCLUDE_DIR}
	"${INCLUDE_DIR}/renderer/VSCore" # VSCore header-only source file
)
target_link_libraries(
	${TARGET_NAME}
	${Qt5_LIBS}
)
