cmake_minimum_required(VERSION 2.8)

project(FaceDetector)

add_definitions(-std=c++11)
add_definitions("-Wall")

set(OpenCV_DIR "/usr/local/opencv/")

if(True)
    find_package(OpenMP)
    if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

include_directories(${OpenCV_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include/ncnn)

link_directories(${OpenCV_DIR}/lib/
                 ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib)
set(OpenCV_LIBS opencv_core 
                opencv_highgui 
                opencv_imgproc 
                opencv_imgcodecs 
                opencv_videoio)

set(SRC
        main.cpp
        FaceDetector.h
        FaceDetector.cpp
        )

add_executable(FaceDetector ${SRC})
target_link_libraries(FaceDetector ncnn ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    if(NCNN_CMAKE_VERBOSE)
        message("Building with OpenMP")
    endif()
    target_link_libraries(FaceDetector PUBLIC OpenMP::OpenMP_CXX)
endif()
