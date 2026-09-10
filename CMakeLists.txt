cmake_minimum_required(VERSION 3.25)
project(mjmlx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MJMLX_BUILD_TESTS "Build tests" ON)
option(MJMLX_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(MJMLX_BUILD_PYTHON "Build Python bindings via nanobind" OFF)
option(MJMLX_BUILD_EXAMPLES "Build example programs" ON)

# ── Tensor backend selection (MLX on Apple, MKX/Vulkan elsewhere) ──
if(NOT DEFINED MJMLX_TENSOR_BACKEND)
    if(APPLE)
        set(MJMLX_TENSOR_BACKEND "MLX")
    else()
        set(MJMLX_TENSOR_BACKEND "MKX")
    endif()
endif()
set(MJMLX_TENSOR_BACKEND "${MJMLX_TENSOR_BACKEND}" CACHE STRING "Tensor backend: MLX or MKX")
set_property(CACHE MJMLX_TENSOR_BACKEND PROPERTY STRINGS MLX MKX)
message(STATUS "MJMLX_TENSOR_BACKEND: ${MJMLX_TENSOR_BACKEND}")

if(MJMLX_TENSOR_BACKEND STREQUAL "MLX")
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

    add_library(tensor_backend INTERFACE)
    target_link_libraries(tensor_backend INTERFACE mlx_imported)
    target_compile_definitions(tensor_backend INTERFACE MJMLX_BACKEND_MLX=1)
    set(MJMLX_RPATH_DIR "${MLX_ROOT}/lib")
else()
    # ── Fetch MKX (Vulkan-based cross-platform MLX-alike) ──────────
    include(FetchContent)
    find_package(Vulkan REQUIRED)
    set(MKX_GIT_TAG "04d9c8407cd9ba75b8098b96636930cab4295580" CACHE STRING "mlx_vulkan commit/tag to fetch")
    FetchContent_Declare(mkx
        GIT_REPOSITORY https://github.com/20niship/mlx_vulkan.git
        GIT_TAG ${MKX_GIT_TAG}
    )
    FetchContent_MakeAvailable(mkx)

    add_library(tensor_backend INTERFACE)
    target_link_libraries(tensor_backend INTERFACE mkx)
    target_compile_definitions(tensor_backend INTERFACE MJMLX_BACKEND_MKX=1)
    set(MJMLX_RPATH_DIR "")
endif()

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
    # Try lib/ subdirectory first (standard install), then root (pip package)
    if(EXISTS "${MUJOCO_ROOT}/lib/libmujoco.dylib")
        set(_MJ_DYLIB_PATH "${MUJOCO_ROOT}/lib/libmujoco.dylib")
    else()
        file(GLOB _MJ_DYLIB "${MUJOCO_ROOT}/libmujoco*.dylib")
        if(_MJ_DYLIB)
            list(GET _MJ_DYLIB 0 _MJ_DYLIB_PATH)
        else()
            message(FATAL_ERROR "Found MuJoCo headers at ${MUJOCO_ROOT} but no libmujoco dylib")
        endif()
    endif()
    # Record the original dylib filename for install_name_tool fixups
    get_filename_component(_MJ_OLD_NAME "${_MJ_DYLIB_PATH}" NAME)
    set_target_properties(mujoco_imported PROPERTIES
        IMPORTED_LOCATION "${_MJ_DYLIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
    )
    message(STATUS "MUJOCO_ROOT: ${MUJOCO_ROOT} -> ${_MJ_DYLIB_PATH}")
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
    src/support.cpp
    src/forward.cpp
    src/smooth_vmap.cpp
    src/constraint_vmap.cpp
    src/solver_vmap.cpp
    src/forward_vmap.cpp
    src/batched.cpp
)

add_library(mjmlx SHARED ${MJMLX_SOURCES})
target_include_directories(mjmlx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(mjmlx PRIVATE tensor_backend mujoco_imported)
set_target_properties(mjmlx PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    BUILD_RPATH "${MJMLX_RPATH_DIR}"
    INSTALL_RPATH "${MJMLX_RPATH_DIR}"
)

# Fix MuJoCo's framework-style install name for pip installs.
# The pip mujoco dylib uses @rpath/mujoco.framework/Versions/A/libmujoco.X.Y.Z.dylib
# but the actual file is a flat dylib. Rewrite the reference to an absolute path.
if(_MJ_DYLIB_PATH)
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

# ── Unified backend library: libmjb ──────────────────────────
add_library(mjb SHARED src/mjb.cpp)
target_include_directories(mjb
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(mjb PRIVATE mjmlx mujoco_imported)
set_target_properties(mjb PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    BUILD_RPATH "${MJMLX_RPATH_DIR}"
    INSTALL_RPATH "${MJMLX_RPATH_DIR}"
)

# Fix MuJoCo dylib reference for libmjb
if(_MJ_DYLIB_PATH AND _MJ_OLD_NAME)
    add_custom_command(TARGET mjb POST_BUILD
        COMMAND install_name_tool -change
            "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
            "$<TARGET_FILE:mjb>"
        COMMENT "Fixing MuJoCo dylib for libmjb"
    )
endif()

install(TARGETS mjb
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# ── Tests ─────────────────────────────────────────────────────
if(MJMLX_BUILD_TESTS)
    enable_testing()
    # ── Active tests (use current batched API) ──────────────────
    add_executable(test_io tests/test_io.cpp)
    target_link_libraries(test_io PRIVATE mjmlx)

    add_executable(test_batched tests/test_batched.cpp)
    target_link_libraries(test_batched PRIVATE mjmlx)

    add_executable(test_batched_diag tests/test_batched_diag.cpp)
    target_link_libraries(test_batched_diag PRIVATE mjmlx mujoco_imported tensor_backend)
    target_include_directories(test_batched_diag PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(test_math_full tests/test_math_full.cpp src/math.cpp)
    target_link_libraries(test_math_full PRIVATE mjmlx mujoco_imported tensor_backend)
    target_include_directories(test_math_full PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(test_linalg_full tests/test_linalg_full.cpp src/smooth_vmap.cpp src/math.cpp)
    target_link_libraries(test_linalg_full PRIVATE mjmlx mujoco_imported tensor_backend)
    target_include_directories(test_linalg_full PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Fix MuJoCo dylib reference for test executables that link mujoco_imported
    if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
        foreach(_test test_math_full test_batched_diag test_linalg_full)
            add_custom_command(TARGET ${_test} POST_BUILD
                COMMAND install_name_tool -change
                    "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
                    "$<TARGET_FILE:${_test}>"
                COMMENT "Fixing MuJoCo dylib for ${_test}"
            )
        endforeach()
    endif()

    # ── CTest entries ─────────────────────────────────────────
    set(MJMLX_TEST_MODEL "" CACHE FILEPATH "Path to humanoid.xml for model-dependent tests")

    add_test(NAME math_full COMMAND test_math_full)
    add_test(NAME linalg_full COMMAND test_linalg_full)
    set_tests_properties(math_full linalg_full PROPERTIES LABELS "correctness")

    # test_metal_synth exercises raw MSL mx::fast::metal_kernel dispatch directly -- MLX-only.
    if(MJMLX_TENSOR_BACKEND STREQUAL "MLX")
        add_executable(test_metal_synth tests/test_metal_synth.cpp)
        target_link_libraries(test_metal_synth PRIVATE mjmlx mujoco_imported tensor_backend)
        target_include_directories(test_metal_synth PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
            add_custom_command(TARGET test_metal_synth POST_BUILD
                COMMAND install_name_tool -change
                    "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
                    "$<TARGET_FILE:test_metal_synth>"
                COMMENT "Fixing MuJoCo dylib for test_metal_synth"
            )
        endif()

        set(MJMLX_SYNTH_MODEL "" CACHE FILEPATH "Path to synth_humanoid.xml for Synth Metal tests")
        if(MJMLX_SYNTH_MODEL)
            add_test(NAME metal_synth COMMAND test_metal_synth ${MJMLX_SYNTH_MODEL})
            set_tests_properties(metal_synth PROPERTIES LABELS "correctness")
        endif()
    endif()

    if(MJMLX_TEST_MODEL)
        add_test(NAME io COMMAND test_io ${MJMLX_TEST_MODEL})
        add_test(NAME batched COMMAND test_batched ${MJMLX_TEST_MODEL} 64 5)
        add_test(NAME batched_diag COMMAND test_batched_diag ${MJMLX_TEST_MODEL})
        set_tests_properties(io batched batched_diag PROPERTIES LABELS "correctness")
    endif()
endif()

# ── Benchmarks ────────────────────────────────────────────────
if(MJMLX_BUILD_BENCHMARKS AND MJMLX_BUILD_TESTS)
    # Baseline benchmark (uses test_utils.h + bench_utils.h + test_models.h)
    add_executable(bench_baseline tests/bench_baseline.cpp)
    target_link_libraries(bench_baseline PRIVATE mjmlx mujoco_imported)
    target_include_directories(bench_baseline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Fix MuJoCo dylib reference for benchmark executables
    if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
        add_custom_command(TARGET bench_baseline POST_BUILD
            COMMAND install_name_tool -change
                "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
                "$<TARGET_FILE:bench_baseline>"
            COMMENT "Fixing MuJoCo dylib for bench_baseline"
        )
    endif()

    # Register benchmark in CTest with "benchmark" label
    if(MJMLX_TEST_MODEL)
        add_test(NAME bench_baseline COMMAND bench_baseline ${MJMLX_TEST_MODEL}
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/history.csv")
    else()
        add_test(NAME bench_baseline COMMAND bench_baseline ""
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/history.csv")
    endif()
    set_tests_properties(bench_baseline PROPERTIES LABELS "benchmark")

    # Synth benchmark exercises raw MSL mx::fast::metal_kernel dispatch directly -- MLX-only.
    if(MJMLX_TENSOR_BACKEND STREQUAL "MLX")
        add_executable(bench_synth tests/bench_synth.cpp)
        target_link_libraries(bench_synth PRIVATE mjmlx mujoco_imported tensor_backend)
        target_include_directories(bench_synth PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
            add_custom_command(TARGET bench_synth POST_BUILD
                COMMAND install_name_tool -change
                    "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
                    "$<TARGET_FILE:bench_synth>"
                COMMENT "Fixing MuJoCo dylib for bench_synth"
            )
        endif()
    endif()

    # Copy benchmark model files to build directory for easy access
    file(COPY tests/models DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tests)
endif()

# ── Python bindings ───────────────────────────────────────────
if(MJMLX_BUILD_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

    # Find nanobind via Python
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(nanobind CONFIG REQUIRED)
    message(STATUS "nanobind found: ${nanobind_DIR}")

    nanobind_add_module(_mjmlx_native
        NB_STATIC STABLE_ABI LTO NOMINSIZE
        NB_DOMAIN mlx
        python/bindings.cpp
    )
    target_link_libraries(_mjmlx_native PRIVATE mjmlx tensor_backend mujoco_imported)
    target_include_directories(_mjmlx_native PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Put the .so next to libmjmlx in the build dir
    set_target_properties(_mjmlx_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    # Fix MuJoCo dylib reference for the Python extension
    if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
        add_custom_command(TARGET _mjmlx_native POST_BUILD
            COMMAND install_name_tool -change
                "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
                "$<TARGET_FILE:_mjmlx_native>"
            COMMENT "Fixing MuJoCo dylib for _mjmlx_native"
        )
    endif()
endif()

# ── Install ───────────────────────────────────────────────────
install(TARGETS mjmlx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/mjmlx DESTINATION include)

