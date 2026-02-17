cmake_minimum_required(VERSION 3.25)
project(mjmlx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MJMLX_BUILD_TESTS "Build tests" ON)
option(MJMLX_BUILD_BENCHMARKS "Build benchmark executables" ON)
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
    BUILD_RPATH "${MLX_ROOT}/lib"
    INSTALL_RPATH "${MLX_ROOT}/lib"
)

# Fix MuJoCo dylib reference for libmjb
if(MUJOCO_PKG_DIR AND _MJ_DYLIB_PATH AND _MJ_OLD_NAME)
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
    add_executable(test_io tests/test_io.cpp)
    target_link_libraries(test_io PRIVATE mjmlx)
    add_executable(test_smooth tests/test_smooth.cpp)
    target_link_libraries(test_smooth PRIVATE mjmlx)
    add_executable(test_forward tests/test_forward.cpp)
    target_link_libraries(test_forward PRIVATE mjmlx)
    add_executable(test_batched tests/test_batched.cpp)
    target_link_libraries(test_batched PRIVATE mjmlx)
    add_executable(test_batched_diag tests/test_batched_diag.cpp)
    target_link_libraries(test_batched_diag PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_batched_diag PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

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

    # Linalg test -- Cholesky and solve (needs internal symbols from smooth_vmap.cpp)
    add_executable(test_linalg_full tests/test_linalg_full.cpp src/smooth_vmap.cpp src/math.cpp)
    target_link_libraries(test_linalg_full PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_linalg_full PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Vmap smooth tests -- com_vel and rne against MuJoCo C reference
    # Uses C API (mjmlx_forward) to run full pipeline, then compares internal Data fields
    add_executable(test_vmap_smooth tests/test_vmap_smooth.cpp)
    target_link_libraries(test_vmap_smooth PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_vmap_smooth PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 1.1: Gravity compensation tests
    add_executable(test_gravcomp tests/test_gravcomp.cpp)
    target_link_libraries(test_gravcomp PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_gravcomp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 1.2: cfrc_ext (per-body contact forces) tests
    add_executable(test_cfrc_ext tests/test_cfrc_ext.cpp)
    target_link_libraries(test_cfrc_ext PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_cfrc_ext PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 1.3: Exclude signature tests
    add_executable(test_exclude tests/test_exclude.cpp)
    target_link_libraries(test_exclude PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_exclude PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 1.4: Model validation warning tests
    add_executable(test_validation tests/test_validation.cpp)
    target_link_libraries(test_validation PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_validation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 1.5: High-DOF model tests and benchmarks
    add_executable(test_high_dof tests/test_high_dof.cpp)
    target_link_libraries(test_high_dof PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_high_dof PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 2.1: Pyramidal friction (condim=3) tests
    add_executable(test_friction_condim3 tests/test_friction_condim3.cpp)
    target_link_libraries(test_friction_condim3 PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_friction_condim3 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 2.2: Condim=4,6 friction tests
    add_executable(test_friction_condim46 tests/test_friction_condim46.cpp)
    target_link_libraries(test_friction_condim46 PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_friction_condim46 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 3.1: BOX collision tests
    add_executable(test_collision_box tests/test_collision_box.cpp)
    target_link_libraries(test_collision_box PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_collision_box PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 3.2: CYLINDER collision tests
    add_executable(test_collision_cylinder tests/test_collision_cylinder.cpp)
    target_link_libraries(test_collision_cylinder PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_collision_cylinder PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Phase 3.3: MESH collision tests (GJK/EPA)
    add_executable(test_collision_mesh tests/test_collision_mesh.cpp)
    target_link_libraries(test_collision_mesh PRIVATE mjmlx mujoco_imported mlx_imported)
    target_include_directories(test_collision_mesh PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Fix MuJoCo dylib reference for test executables that link mujoco_imported
    if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
        foreach(_test test_io_full test_forward_full test_physics_full test_solver_full test_collision_full test_math_full test_batched_diag test_linalg_full test_vmap_smooth test_gravcomp test_cfrc_ext test_exclude test_validation test_high_dof test_friction_condim3 test_friction_condim46 test_collision_box test_collision_cylinder test_collision_mesh)
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

    # Self-contained tests (no model path needed)
    add_test(NAME math_full COMMAND test_math_full)
    set_tests_properties(math_full PROPERTIES LABELS "correctness")

    # Tests that optionally use a model path
    if(MJMLX_TEST_MODEL)
        add_test(NAME io_full COMMAND test_io_full)
        add_test(NAME forward_full COMMAND test_forward_full)
        add_test(NAME physics_full COMMAND test_physics_full)
        add_test(NAME solver_full COMMAND test_solver_full ${MJMLX_TEST_MODEL})
        add_test(NAME collision_full COMMAND test_collision_full)
        add_test(NAME linalg_full COMMAND test_linalg_full ${MJMLX_TEST_MODEL})
        add_test(NAME batched_diag COMMAND test_batched_diag ${MJMLX_TEST_MODEL})
        add_test(NAME batched COMMAND test_batched ${MJMLX_TEST_MODEL} 64 5)
        add_test(NAME vmap_smooth COMMAND test_vmap_smooth)
    else()
        add_test(NAME io_full COMMAND test_io_full)
        add_test(NAME forward_full COMMAND test_forward_full)
        add_test(NAME physics_full COMMAND test_physics_full)
        add_test(NAME solver_full COMMAND test_solver_full)
        add_test(NAME collision_full COMMAND test_collision_full)
        add_test(NAME linalg_full COMMAND test_linalg_full)
        add_test(NAME vmap_smooth COMMAND test_vmap_smooth)
    endif()

    # Self-contained Phase 1 tests (no model path needed)
    add_test(NAME gravcomp COMMAND test_gravcomp)
    add_test(NAME cfrc_ext COMMAND test_cfrc_ext)
    add_test(NAME exclude COMMAND test_exclude)
    add_test(NAME validation COMMAND test_validation)
    add_test(NAME high_dof COMMAND test_high_dof)
    add_test(NAME friction_condim3 COMMAND test_friction_condim3)
    add_test(NAME friction_condim46 COMMAND test_friction_condim46)
    add_test(NAME collision_box COMMAND test_collision_box)
    add_test(NAME collision_cylinder COMMAND test_collision_cylinder)
    add_test(NAME collision_mesh COMMAND test_collision_mesh)
    set_tests_properties(gravcomp cfrc_ext exclude validation high_dof friction_condim3 friction_condim46 collision_box collision_cylinder collision_mesh PROPERTIES LABELS "correctness")

    # Label all correctness tests
    set(_correctness_tests io_full forward_full physics_full solver_full collision_full linalg_full vmap_smooth)
    if(MJMLX_TEST_MODEL)
        list(APPEND _correctness_tests batched_diag batched)
    endif()
    foreach(_ct ${_correctness_tests})
        set_tests_properties(${_ct} PROPERTIES LABELS "correctness")
    endforeach()
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
    target_link_libraries(_mjmlx_native PRIVATE mjmlx mlx_imported)
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
