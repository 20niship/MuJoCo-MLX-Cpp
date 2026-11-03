// MJMLX_BUILD_CAMERA=OFF用のスタブ。カメラ無しでもビルド・学習が動くようにし、create時に明確なエラーを返す。

#include "internal.h"
#include "mjmlx/mjmlx.h"

#include <cstdio>

namespace mjmlx {
void camera_prepare_device() {}
} // namespace mjmlx

extern "C" {

MJMLX_API MjmlxCameraRenderer* mjmlx_camera_create(MjmlxBatchedSim*, int, int) {
    fprintf(stderr, "mjmlx_camera_create error: camera support not built (MJMLX_BUILD_CAMERA=OFF)\n");
    return nullptr;
}
MJMLX_API void mjmlx_camera_free(MjmlxCameraRenderer*) {}
MJMLX_API int mjmlx_camera_count(const MjmlxCameraRenderer*) { return 0; }
MJMLX_API int mjmlx_camera_draw_call_count(const MjmlxCameraRenderer*) { return 0; }
MJMLX_API int mjmlx_camera_render(MjmlxCameraRenderer*) { return -1; }
MJMLX_API int mjmlx_camera_get_rgb(MjmlxCameraRenderer*, uint8_t*) { return -1; }
MJMLX_API int mjmlx_camera_get_depth(MjmlxCameraRenderer*, float*) { return -1; }
MJMLX_API int mjmlx_camera_gpu_supported(const MjmlxCameraRenderer*) { return 0; }
MJMLX_API int mjmlx_camera_gpu_copy(MjmlxCameraRenderer*) { return -1; }
MJMLX_API int mjmlx_camera_gpu_export_fd(MjmlxCameraRenderer*, int, uint64_t*) { return -1; }

} // extern "C"
