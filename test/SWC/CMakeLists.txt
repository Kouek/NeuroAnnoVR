set(TARGET_NAME "testSWC")

add_executable(
	${TARGET_NAME}
	"main.cpp"
)
target_include_directories(
	${TARGET_NAME}
	PRIVATE
	${INCLUDE_DIR}
)
target_link_libraries(
	${TARGET_NAME}
	PRIVATE
	"glm::glm"
)
