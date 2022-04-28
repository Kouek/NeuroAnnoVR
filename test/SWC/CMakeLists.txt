set(TARGET_NAME "testSWC")

add_executable(
	${TARGET_NAME}
	"main.cpp"
	"GLView.h" # Necessary! Or it won't be MOC
	"${THIRDPARTY_DIR}/glad/src/glad.c"
)
target_include_directories(
	${TARGET_NAME}
	PRIVATE
	${INCLUDE_DIR}
)
target_link_libraries(
	${TARGET_NAME}
	${Qt5_LIBS}
	"glm::glm"
)
