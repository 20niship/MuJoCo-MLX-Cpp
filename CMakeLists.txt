cmake_minimum_required(VERSION 3.25)

# 未指定時はReleaseをデフォルトにする(素のcmake -B buildだと最適化なしになり、batched等のCPU側オーバーヘッドがベンチマークで数十%重く出る)。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(mjmlx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MJMLX_BUILD_TESTS "Build tests" ON)
option(MJMLX_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(MJMLX_BUILD_PYTHON "Build Python bindings via nanobind" OFF)
option(MJMLX_BUILD_EXAMPLES "Build example programs" ON)

# ── Tensor backend: MKX (Vulkan版MLX相当) ──
# git submoduleとしてvendor化(FetchContent+MKX_GIT_TAGのpin取り違え事故対策、issue #14)。
find_package(Vulkan REQUIRED)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/mlx_vulkan/CMakeLists.txt")
    message(FATAL_ERROR "extern/mlx_vulkan submodule not found. Run: git submodule update --init --recursive")
endif()
# mkx's own CMakeLists builds a plain STATIC lib (no PIC); linking that into our .so needs -fPIC on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(extern/mlx_vulkan)

add_library(tensor_backend INTERFACE)
target_link_libraries(tensor_backend INTERFACE mkx)
set(MJMLX_RPATH_DIR "")

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
        file(GLOB _MJ_DYLIB "${MUJOCO_ROOT}/libmujoco*.dylib" "${MUJOCO_ROOT}/libmujoco*.so*")
        if(_MJ_DYLIB)
            list(GET _MJ_DYLIB 0 _MJ_DYLIB_PATH)
        else()
            message(FATAL_ERROR "Found MuJoCo headers at ${MUJOCO_ROOT} but no libmujoco dylib/so")
        endif()
    endif()
    set_target_properties(mujoco_imported PROPERTIES
        IMPORTED_LOCATION "${_MJ_DYLIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
    )
    message(STATUS "MUJOCO_ROOT: ${MUJOCO_ROOT} -> ${_MJ_DYLIB_PATH}")
elseif(MUJOCO_PKG_DIR AND EXISTS "${MUJOCO_PKG_DIR}/include/mujoco/mujoco.h")
    add_library(mujoco_imported SHARED IMPORTED)
    file(GLOB _MJ_DYLIB "${MUJOCO_PKG_DIR}/libmujoco*.dylib" "${MUJOCO_PKG_DIR}/libmujoco*.so*")
    if(_MJ_DYLIB)
        list(GET _MJ_DYLIB 0 _MJ_DYLIB_PATH)
    else()
        file(GLOB _MJ_DYLIB "${MUJOCO_PKG_DIR}/*.dylib" "${MUJOCO_PKG_DIR}/*.so*")
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
if(APPLE AND _MJ_DYLIB_PATH)
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

    add_executable(test_gpu_vs_cpu tests/test_gpu_vs_cpu.cpp)
    target_link_libraries(test_gpu_vs_cpu PRIVATE mjmlx mujoco_imported tensor_backend)
    target_include_directories(test_gpu_vs_cpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(test_gpu_accuracy tests/test_gpu_accuracy.cpp)
    target_link_libraries(test_gpu_accuracy PRIVATE mjmlx mujoco_imported tensor_backend)
    target_include_directories(test_gpu_accuracy PRIVATE
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

    # Self-contained (embeds its own XML fixtures) GPU-vs-CPU batched collision parity test --
    # no MJMLX_TEST_MODEL needed, so it always runs in CI.
    add_executable(test_batched_collision_primitives tests/test_batched_collision_primitives.cpp)
    target_link_libraries(test_batched_collision_primitives PRIVATE mjmlx mujoco_imported tensor_backend)
    target_include_directories(test_batched_collision_primitives PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Fix MuJoCo dylib reference for test executables that link mujoco_imported
    if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
        foreach(_test test_math_full test_batched_diag test_gpu_accuracy test_gpu_vs_cpu test_linalg_full test_batched_collision_primitives)
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
    add_test(NAME batched_collision_primitives COMMAND test_batched_collision_primitives)
    set_tests_properties(math_full linalg_full batched_collision_primitives PROPERTIES LABELS "correctness")

    # GPU(MKX) vs CPU(MuJoCo C)のバッチ比較(旧test_metal_synthの代替)。引数は[tol_1step tol_10step tol_100step](qvelは10倍)。
    # humanoidは厳密に一致する。high_dof_tree/contact_stressはGPU-CPU間に既知の乖離があり、現状値を少し超える閾値でのリグレッション検知用(乖離を直したら締める)。
    set(_M "${CMAKE_CURRENT_SOURCE_DIR}/tests/models")
    add_test(NAME gpu_vs_cpu_humanoid COMMAND test_gpu_vs_cpu ${_M}/humanoid.xml 1e-4 1e-3 1.0)
    add_test(NAME gpu_vs_cpu_high_dof COMMAND test_gpu_vs_cpu ${_M}/high_dof_tree.xml 0.5 0.5 3.0)
    add_test(NAME gpu_vs_cpu_contact_stress COMMAND test_gpu_vs_cpu ${_M}/contact_stress.xml 1e-2 5e-2 0.5)
    set_tests_properties(gpu_vs_cpu_humanoid gpu_vs_cpu_high_dof gpu_vs_cpu_contact_stress PROPERTIES LABELS "correctness")

    # mesh付き実ロボットMJCFの軌道比較(関節角・body位置/姿勢のCPU-GPU差)。モデルは `just fetch-models` で benchmarks/models/external/ に取得(git管理外)。
    # 未取得なら登録しない。LABELS=modelsなのでCIの `just check-ci` には含まれず、`just check-models` で実行する。
    # 閾値は実測(NVIDIA)の約2倍。接触(着地)を含む脚ロボットはCPU/GPUの接触解の差で発散するため、自由落下(接触前)は厳しく、着地後は現状値を少し超える緩い閾値で回帰検知のみ行う(乖離を直したら締める)。
    set(_EXT "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/models/external")
    function(mjmlx_add_model_test name model)
        if(EXISTS "${_EXT}/${model}")
            add_test(NAME model_gpu_vs_cpu_${name} COMMAND test_gpu_vs_cpu "${_EXT}/${model}" ${ARGN})
            set_tests_properties(model_gpu_vs_cpu_${name} PROPERTIES LABELS "models" TIMEOUT 1800)
        endif()
    endfunction()
    set(_MG "mujoco_menagerie")
    mjmlx_add_model_test(piper ${_MG}/agilex_piper/scene.xml --seconds 4 --ctrl sin --bodies base_link,link6,link8 --tol-joint 0.08 --tol-pos 0.08 --tol-rot 0.01)
    mjmlx_add_model_test(openarm openarm_mujoco/v1/scene.xml --seconds 4 --ctrl sin --bodies openarm_left_link7,openarm_left_hand_tcp,openarm_right_hand_tcp --tol-joint 0.25 --tol-pos 0.06 --tol-rot 0.4)
    mjmlx_add_model_test(go2_freefall ${_MG}/unitree_go2/scene.xml --seconds 0.6 --lift 3 --ctrl sin --bodies base,FL_calf,FR_calf --tol-joint 1e-3 --tol-pos 1e-2 --tol-rot 5e-3)
    mjmlx_add_model_test(go2_landing ${_MG}/unitree_go2/scene.xml --seconds 1 --lift 0.3 --ctrl sin --bodies base,FL_calf,FR_calf --tol-joint 0.5 --tol-pos 0.15 --tol-rot 0.6)
    mjmlx_add_model_test(g1_freefall ${_MG}/unitree_g1/scene.xml --seconds 0.6 --lift 3 --ctrl sin --bodies pelvis,left_ankle_roll_link,right_wrist_yaw_link --tol-joint 1e-3 --tol-pos 1e-2 --tol-rot 2e-2)
    mjmlx_add_model_test(g1_landing ${_MG}/unitree_g1/scene.xml --seconds 1 --lift 0.1 --ctrl sin --bodies pelvis,left_ankle_roll_link,right_wrist_yaw_link --tol-joint 0.1 --tol-pos 0.15 --tol-rot 0.5)
    mjmlx_add_model_test(h1_freefall ${_MG}/unitree_h1/scene.xml --seconds 0.6 --lift 3 --ctrl sin --bodies pelvis,left_ankle_link,right_elbow_link --tol-joint 0.35 --tol-pos 0.05 --tol-rot 0.15)

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

    # Numpy-facing RL bindings (mjmlx.h only, no mlx.core NB_DOMAIN sharing -- see python/rl_bindings.cpp)
    nanobind_add_module(_mjmlx_rl_native
        NB_STATIC STABLE_ABI LTO NOMINSIZE
        python/rl_bindings.cpp
    )
    target_link_libraries(_mjmlx_rl_native PRIVATE mjmlx)
    target_include_directories(_mjmlx_rl_native PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    set_target_properties(_mjmlx_rl_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    if(_MJ_OLD_NAME AND _MJ_DYLIB_PATH)
        add_custom_command(TARGET _mjmlx_rl_native POST_BUILD
            COMMAND install_name_tool -change
                "${_MJ_OLD_NAME}" "${_MJ_DYLIB_PATH}"
                "$<TARGET_FILE:_mjmlx_rl_native>"
            COMMENT "Fixing MuJoCo dylib for _mjmlx_rl_native"
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

