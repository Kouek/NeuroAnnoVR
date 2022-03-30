# renderer build
add_subdirectory("renderer")

# demo build
message(STATUS "${PROJECT_NAME}_BUILD_DEMO: ${${PROJECT_NAME}_BUILD_DEMO}")
if(${${PROJECT_NAME}_BUILD_DEMO})
	add_subdirectory("QtGLDemo")
endif()

add_subdirectory("NeuroAnnoVR")
