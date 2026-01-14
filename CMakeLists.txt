cmake_minimum_required(VERSION 3.14)
project(DigitApp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    LinAlg
    GIT_REPOSITORY https://github.com/YoussefMohamedIbrahim/LinAlg.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(LinAlg)

add_executable(DigitApp src/main.cpp)

target_link_libraries(DigitApp PRIVATE LinAlg)

target_include_directories(DigitApp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)