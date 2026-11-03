"""カメラのRGB/depthをCPUを経由せずGPU上のtorchテンソルとして取る(Vulkan->CUDA external memory interop、Linuxのみ)。

使い方: cam = native.create_camera(...); gcam = CudaCamera.try_create(cam)  # 非対応ならNone -> cam.rgb()/depth()のCPU読み出しを使う
        cam.render(); rgb, depth = gcam.read()   # rgb: uint8 (env,cam,H,W,4)のRGBA, depth: float32 (env,cam,H,W)
テンソルは常に同じメモリを指し、次のread()で上書きされる(必要なら呼び出し側でclone)。CUDAランタイムはtorchが読み込み済みのlibcudartをctypesで使う(ビルド時のCUDA依存なし)。
"""

import ctypes
import os

_RGBA, _DEPTH = 0, 1


def _cudart():
    import torch

    torch.zeros(1, device="cuda")
    for line in open("/proc/self/maps"):
        if "libcudart.so" in line:
            return ctypes.CDLL(line.split()[-1])
    raise RuntimeError("libcudart is not loaded")


class _HandleDesc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("_align", ctypes.c_int), ("fd", ctypes.c_int), ("_pad", ctypes.c_byte * 12), ("size", ctypes.c_ulonglong), ("flags", ctypes.c_uint)]


class _BufDesc(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_ulonglong), ("size", ctypes.c_ulonglong), ("flags", ctypes.c_uint)]


class _Holder:
    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": typestr, "data": (ptr, False), "version": 3}


class CudaCamera:
    @classmethod
    def try_create(cls, cam):
        """interopが使えなければNone(mac、MJMLX_BUILD_CAMERA=OFF、VK_KHR_external_memory_fd無し、CUDA無し等)。"""
        if os.name != "posix" or not cam.gpu_supported:
            return None
        try:
            return cls(cam)
        except Exception:
            return None

    def __init__(self, cam):
        import torch

        self.cam, self._rt = cam, _cudart()
        n = (cam.num_envs, cam.num_cameras, cam.height, cam.width)
        self._shapes = (n + (4,), n)
        self._tensors = None
        self._torch = torch

    def _import(self):
        self.cam.gpu_copy()
        out = []
        for which, typestr in ((_RGBA, "|u1"), (_DEPTH, "<f4")):
            fd, size = self.cam.gpu_export_fd(which)
            d = _HandleDesc(type=1, fd=fd, size=size)  # cudaExternalMemoryHandleTypeOpaqueFd。fdの所有権はCUDAへ移る
            mem = ctypes.c_void_p()
            if self._rt.cudaImportExternalMemory(ctypes.byref(mem), ctypes.byref(d)) != 0:
                raise RuntimeError("cudaImportExternalMemory failed")
            nbytes = 1
            for s in self._shapes[which]:
                nbytes *= s
            nbytes *= 1 if which == _RGBA else 4
            ptr = ctypes.c_void_p()
            if self._rt.cudaExternalMemoryGetMappedBuffer(ctypes.byref(ptr), mem, ctypes.byref(_BufDesc(0, nbytes, 0))) != 0:
                raise RuntimeError("cudaExternalMemoryGetMappedBuffer failed")
            out.append(self._torch.as_tensor(_Holder(ptr.value, self._shapes[which], typestr), device="cuda"))
        self._tensors = tuple(out)

    def read(self):
        """直前のcam.render()の結果を (rgba, depth) のGPUテンソルで返す。Vulkan側のコピー完了後に返るので追加の同期は不要。"""
        if self._tensors is None:
            self._import()
        else:
            self.cam.gpu_copy()
        return self._tensors
