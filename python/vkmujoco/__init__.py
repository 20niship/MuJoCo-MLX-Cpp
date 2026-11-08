"""vkmujoco: MuJoCo-compatible batched GPU physics on Vulkan."""

from ._core import BatchedSim, Camera, Model, create_batched, create_camera, load_model, version

__version__ = version()
__all__ = ["BatchedSim", "Camera", "Model", "__version__", "create_batched", "create_camera", "load_model", "version"]
