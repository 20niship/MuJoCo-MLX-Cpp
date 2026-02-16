# nanobind Python extension for MuJoCo-MLX-Cpp
#
# Builds _mjmlx_native.cpython-*.so which exposes the C++ physics pipeline
# as Python mx.array API. Uses NB_DOMAIN "mlx" so that mlx::core::array
# objects are seamlessly shared with the mlx Python package.

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind (installed via pip install nanobind)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_mjmlx_native
    NB_DOMAIN mlx
    python/bindings.cpp
)

# Link against our core library + MLX
target_link_libraries(_mjmlx_native PRIVATE mjmlx mlx_imported)
target_include_directories(_mjmlx_native PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Install the extension next to libmjmlx
install(TARGETS _mjmlx_native LIBRARY DESTINATION lib)
