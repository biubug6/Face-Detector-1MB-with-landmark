set(NCNN_OPENMP ON)
set(NCNN_VULKAN OFF)

if(NCNN_OPENMP)
    find_package(OpenMP)
endif()

if(NCNN_VULKAN)
    find_package(Vulkan REQUIRED)

    add_library(Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan PROPERTIES IMPORTED_LOCATION ${Vulkan_LIBRARY})
    set_target_properties(Vulkan PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${Vulkan_INCLUDE_DIR})
endif()

include(${CMAKE_CURRENT_LIST_DIR}/ncnn.cmake)
