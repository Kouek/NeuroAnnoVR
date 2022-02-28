# This demo help you examine the completeness of the enviroment
# and test the performance of the Neuron Renderer.

set(TARGET_NAME "QtGLDemo")

add_executable(
	${TARGET_NAME}
	"DemoWindow.ui"
	"main.cpp"
	"DemoWindow.cpp"
	"${THIRDPARTY_DIR}/glad/src/glad.c"
)
target_link_libraries(
	${TARGET_NAME}
	${Qt5_LIBS}
	"SubregionRenderer"
)
