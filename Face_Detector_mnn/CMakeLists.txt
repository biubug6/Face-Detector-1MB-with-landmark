cmake_minimum_required(VERSION 3.10)
project(retinaface-mnn)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(
	mnn/include/MNN
	mnn/include
        ./
)

link_directories(mnn/lib)

add_executable(retinaface-mnn main.cpp common.cpp reatinaface.cpp )
target_link_libraries(retinaface-mnn MNN ${OpenCV_LIBS})
