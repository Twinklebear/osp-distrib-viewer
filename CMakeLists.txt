## ======================================================================== ##
## Copyright 2009-2018 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

cmake_minimum_required(VERSION 3.10)
project(ospDistribViewer)

find_package(MPI REQUIRED)
find_package(ospray REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.2 REQUIRED)

include_directories(${OSPRAY_INCLUDE_DIRS})

add_subdirectory(gensv)
add_subdirectory(imgui)

add_executable(ospDistribViewer ospDistribViewer.cpp
	arcball.cpp
	imgui_impl_glfw_gl3.cpp)
target_include_directories(ospDistribViewer PUBLIC
	$<BUILD_INTERFACE:${OSPRAY_INCLUDE_DIRS}>)
target_link_libraries(ospDistribViewer PUBLIC
	gensv
	imgui
	MPI::MPI_CXX
	OpenGL::GL
	glfw
	${OSPRAY_LIBRARIES}
	${OSPRAY_COMMON_LIBRARY}
	${OSPRAY_MPI_LIBRARIES}
	${VTK_LIBRARIES})

