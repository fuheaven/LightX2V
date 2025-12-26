import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("gcu")
class EnflameGcuDevice:
    """
    Enflame GCU Device implementation for LightX2V.

    Enflame GCU uses torch_gcu which provides CUDA-compatible APIs.
    Most PyTorch operations work transparently through the GCU backend.
    """

    name = "gcu"

    @staticmethod
    def is_available() -> bool:
        """
        Check if Enflame GCU is available.

        Uses torch_gcu.gcu.is_available() to check device availability.

        Returns:
            bool: True if Enflame GCU is available
        """
        try:
            import torch_gcu
            return torch_gcu.gcu.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        """
        Get the device type string.

        Returns "gcu" for Enflame GCU device. This allows getattr(torch, AI_DEVICE)
        to work correctly (torch.gcu) and torch.device(AI_DEVICE) to work with GCU.

        Returns:
            str: "gcu" for GCU device
        """
        return "gcu"

    @staticmethod
    def init_parallel_env():
        """
        Initialize distributed parallel environment for Enflame GCU.

        Uses ECCL (Enflame Collective Communication Library) which is
        compatible with NCCL APIs for multi-GPU communication.
        """
        # ECCL is compatible with NCCL backend
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())

