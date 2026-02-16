cmake_minimum_required(VERSION 3.25)
project(mjmlx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MJMLX_BUILD_TESTS "Build tests" ON)
option(MJMLX_BUILD_PYTHON "Build Python bindings via nanobind" OFF)
option(MJMLX_BUILD_EXAMPLES "Build example programs" ON)

# ── Find MLX ──────────────────────────────────────────────────
set(MLX_ROOT "" CACHE PATH "Path to MLX installation (e.g. site-packages/mlx)")

if(NOT MLX_ROOT)
    execute_process(
        COMMAND python3 -c "import mlx, os; print(os.path.dirname(mlx.__file__))"
        OUTPUT_VARIABLE MLX_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

if(NOT MLX_ROOT OR NOT EXISTS "${MLX_ROOT}/include/mlx/mlx.h")
    message(FATAL_ERROR
        "Cannot find MLX. Set -DMLX_ROOT=/path/to/mlx or ensure 'pip install mlx' is in PATH.\n"
        "  Tried: ${MLX_ROOT}")
endif()

message(STATUS "MLX_ROOT: ${MLX_ROOT}")

add_library(mlx_imported SHARED IMPORTED)
set_target_properties(mlx_imported PROPERTIES
    IMPORTED_LOCATION "${MLX_ROOT}/lib/libmlx.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${MLX_ROOT}/include"
)

# ── Find MuJoCo ──────────────────────────────────────────────
set(MUJOCO_ROOT "" CACHE PATH "Path to MuJoCo installation")

if(NOT MUJOCO_ROOT)
    # Try common locations
    foreach(candidate
        "/usr/local"
        "/opt/homebrew"
        "$ENV{HOME}/.mujoco/mujoco-3.0.0"
        "$ENV{HOME}/miniconda3")
        if(EXISTS "${candidate}/include/mujoco/mujoco.h")
            set(MUJOCO_ROOT "${candidate}")
            break()
        endif()
    endforeach()
endif()

# Also try Python mujoco package (pip install mujoco puts headers inside the package)
if(NOT MUJOCO_ROOT OR NOT EXISTS "${MUJOCO_ROOT}/include/mujoco/mujoco.h")
    # Try to find via pip's mujoco package
    execute_process(
        COMMAND pip show -f mujoco
        OUTPUT_VARIABLE _PIP_SHOW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND pip show mujoco
        OUTPUT_VARIABLE _PIP_INFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    # Extract Location from pip show output
    string(REGEX MATCH "Location: ([^\n]+)" _LOC_MATCH "${_PIP_INFO}")
    if(_LOC_MATCH)
        string(REGEX REPLACE "Location: " "" _PIP_LOC "${_LOC_MATCH}")
        set(MUJOCO_PKG_DIR "${_PIP_LOC}/mujoco")
        if(EXISTS "${MUJOCO_PKG_DIR}/include/mujoco/mujoco.h")
            message(STATUS "MuJoCo found via pip at: ${MUJOCO_PKG_DIR}")
        else()
            set(MUJOCO_PKG_DIR "")
        endif()
    endif()
endif()

# Create imported target for MuJoCo
if(MUJOCO_ROOT AND EXISTS "${MUJOCO_ROOT}/include/mujoco/mujoco.h")
    add_library(mujoco_imported SHARED IMPORTED)
    set_target_properties(mujoco_imported PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_ROOT}/lib/libmujoco.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
    )
    message(STATUS "MUJOCO_ROOT: ${MUJOCO_ROOT}")
elseif(MUJOCO_PKG_DIR AND EXISTS "${MUJOCO_PKG_DIR}/include/mujoco/mujoco.h")
    add_library(mujoco_imported SHARED IMPORTED)
    file(GLOB _MJ_DYLIB "${MUJOCO_PKG_DIR}/libmujoco*.dylib")
    if(_MJ_DYLIB)
        list(GET _MJ_DYLIB 0 _MJ_DYLIB_PATH)
    else()
        file(GLOB _MJ_DYLIB "${MUJOCO_PKG_DIR}/*.dylib")
        if(_MJ_DYLIB)
            list(GET _MJ_DYLIB 0 _MJ_DYLIB_PATH)
        endif()
    endif()
    set_target_properties(mujoco_imported PROPERTIES
        IMPORTED_LOCATION "${_MJ_DYLIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_PKG_DIR}/include"
    )
    message(STATUS "MuJoCo (pip): ${MUJOCO_PKG_DIR} -> ${_MJ_DYLIB_PATH}")
else()
    message(FATAL_ERROR
        "Cannot find MuJoCo. Set -DMUJOCO_ROOT=/path/to/mujoco or ensure 'pip install mujoco' is in PATH.")
endif()

# ── Core library: libmjmlx ───────────────────────────────────
set(MJMLX_SOURCES
    src/types.cpp
    src/io.cpp
    src/math.cpp
    src/scan.cpp
    src/passive.cpp
    src/support.cpp
    src/smooth.cpp
    src/collision.cpp
    src/constraint.cpp
    src/solver.cpp
    src/forward.cpp
    src/smooth_vmap.cpp
    src/constraint_vmap.cpp
    src/solver_vmap.cpp
    src/forward_vmap.cpp
    src/batched.cpp
    src/nn.cpp
    src/ppo.cpp
)

add_library(mjmlx SHARED ${MJMLX_SOURCES})
target_include_directories(mjmlx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(mjmlx PRIVATE mlx_imported mujoco_imported)
set_target_properties(mjmlx PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    BUILD_RPATH "${MLX_ROOT}/lib"
    INSTALL_RPATH "${MLX_ROOT}/lib"
)

# Fix MuJoCo's framework-style install name for pip installs.
# The pip mujoco dylib uses @rpath/mujoco.framework/Versions/A/libmujoco.X.Y.Z.dylib
# but the actual file is a flat dylib. Rewrite the reference to an absolute path.
if(MUJOCO_PKG_DIR AND _MJ_DYLIB_PATH)
    execute_process(
        COMMAND otool -D "${_MJ_DYLIB_PATH}"
        OUTPUT_VARIABLE _MJ_INSTALL_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Extract the framework-style install name (second line of otool -D output)
    string(REGEX MATCH "@rpath[^\n]+" _MJ_OLD_NAME "${_MJ_INSTALL_NAME}")
    if(_MJ_OLD_NAME)
        add_custom_command(TARGET mjmlx POST_BUILD
            COMMAND install_name_tool -change
                "${_MJ_OLD_NAME}"
                "${_MJ_DYLIB_PATH}"
                "$<TARGET_FILE:mjmlx>"
            COMMENT "Fixing MuJoCo dylib reference: ${_MJ_OLD_NAME} -> ${_MJ_DYLIB_PATH}"
        )
    endif()
endif()

# ── Tests ─────────────────────────────────────────────────────
if(MJMLX_BUILD_TESTS)
    enable_testing()
    add_executable(test_io tests/test_io.cpp)
    target_link_libraries(test_io PRIVATE mjmlx)
    add_executable(test_smooth tests/test_smooth.cpp)
    target_link_libraries(test_smooth PRIVATE mjmlx)
    add_executable(test_forward tests/test_forward.cpp)
    target_link_libraries(test_forward PRIVATE mjmlx)
    add_executable(test_batched tests/test_batched.cpp)
    target_link_libraries(test_batched PRIVATE mjmlx)

    # Full test suite (ported from Python 228 tests)
    # All full tests need mjmlx + mujoco + internal headers
    # Tests using C API + MuJoCo C reference
    foreach(_test test_io_full test_forward_full test_physics_full test_solver_full test_collision_full)
        add_executable(${_test} tests/${_test}.cpp)
        target_link_libraries(${_test} PRIVATE mjmlx mujoco_imported)
        target_include_directories(${_test} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
    endforeach()

    # Math test needs internal symbols -- link math.cpp directly
    add_executable(test_math_full tests/test_math_full.cpp src/math.cpp)
    target_link_libraries(test_math_full PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_math_full PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Fix MuJoCo dylib reference for test executables that link mujoco_imported
    if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
        foreach(_test test_io_full test_forward_full test_physics_full test_solver_full test_collision_full test_math_full)
            add_custom_command(TARGET ${_test} POST_BUILD
                COMMAND install_name_tool -change
                    "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
                    "$<TARGET_FILE:${_test}>"
                COMMENT "Fixing MuJoCo dylib for ${_test}"
            )
        endforeach()
    endif()
endif()

# ── Python bindings ───────────────────────────────────────────
if(MJMLX_BUILD_PYTHON)
    # add_subdirectory(python)
endif()

# ── Install ───────────────────────────────────────────────────
install(TARGETS mjmlx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/mjmlx DESTINATION include)
