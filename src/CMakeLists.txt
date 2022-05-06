# renderer build
add_subdirectory("renderer")

# demo build
if(${${PROJECT_NAME}_BUILD_DEMO})
	add_subdirectory("QtGLDemo")
endif()

add_subdirectory("NeuroAnnoVR")
