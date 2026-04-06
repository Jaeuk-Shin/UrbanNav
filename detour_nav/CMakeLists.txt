cmake_minimum_required(VERSION 3.15)
project(detour_nav LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Fetch recastnavigation ──────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    recastnavigation
    GIT_REPOSITORY https://github.com/recastnavigation/recastnavigation.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
# Only build Detour (we don't need Recast mesh baking or demo apps)
set(RECASTNAVIGATION_DEMO OFF CACHE BOOL "" FORCE)
set(RECASTNAVIGATION_TESTS OFF CACHE BOOL "" FORCE)
set(RECASTNAVIGATION_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(recastnavigation)

# ── pybind11 ────────────────────────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── Build the extension module ──────────────────────────────────────
pybind11_add_module(detour_nav src/bindings.cpp)
target_link_libraries(detour_nav PRIVATE Detour)
target_include_directories(detour_nav PRIVATE
    ${recastnavigation_SOURCE_DIR}/Detour/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
