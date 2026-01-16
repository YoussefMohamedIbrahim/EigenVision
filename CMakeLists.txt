cmake_minimum_required(VERSION 3.14)
project(EigenVisionPy LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
    LinAlg
    GIT_REPOSITORY https://github.com/YoussefMohamedIbrahim/LinAlg.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(LinAlg)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# Create the Python Module
pybind11_add_module(eigenvision src/bindings.cpp)

target_link_libraries(eigenvision PRIVATE LinAlg)
target_include_directories(eigenvision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(MODELS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models")
file(MAKE_DIRECTORY ${MODELS_PATH})
target_compile_definitions(eigenvision PRIVATE "MODELS_DIR=\"${MODELS_PATH}\"")

set_target_properties(eigenvision PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)