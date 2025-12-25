import torch
from loguru import logger

from lightx2v_platform.ops.mm.template import MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

# Try to import torchao for quantized matrix multiplication
try:
    from torchao.quantization.utils import quant_int8_per_token_matmul as torchao_int8_gemm
    from torchao.quantization.utils import quantize_activation_per_token_absmax as torchao_int8_quant
except ImportError:
    try:
        from torchao.quantization.utils import _quant_int8_per_token_matmul as torchao_int8_gemm
        from torchao.quantization.utils import _quantize_activation_per_token_absmax as torchao_int8_quant
    except ImportError:
        torchao_int8_gemm, torchao_int8_quant = None, None
        logger.warning("torchao not available, will use standard PyTorch matmul")


@PLATFORM_MM_WEIGHT_REGISTER("mm_enflame_gcu")
class MMWeightEnflameGcu(MMWeightQuantTemplate):
    """
    Enflame GCU Matrix Multiplication implementation.

    Uses torchao for quantized matrix multiplication when available.
    Falls back to standard PyTorch matmul with dequantization if torchao is not available.

    Key compatibility notes for GCU:
    - GCU does not support float64, use float32
    - GCU does not support int64/uint64, use int32
    - All operations use standard PyTorch ops compatible with GCU
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
        )
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_int8_perchannel_sym_torchao
        self.use_torchao = torchao_int8_gemm is not None

    def act_quant_int8_perchannel_sym_torchao(self, x):
        """
        Quantize activation using torchao (per-token absmax quantization).

        Args:
            x: Input tensor to quantize
        Returns:
            quantized_tensor: Quantized tensor (int8)
            scale: Scale tensor (float32)
        """
        if torchao_int8_quant is None:
            logger.warning("torchao not available, using fallback quantization")
            # Fallback: return original tensor and scale of 1.0
            return x, torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)

        input_tensor_quant, input_tensor_scale = torchao_int8_quant(x)
        return input_tensor_quant, input_tensor_scale

    def apply(self, input_tensor):
        """
        Apply quantized matrix multiplication.

        Args:
            input_tensor: Input tensor [M, K]
        Returns:
            output_tensor: Output tensor [M, N]
        """
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)

        if self.use_torchao and torchao_int8_gemm is not None:
            # Use torchao's optimized quantized matmul
            output_tensor = torchao_int8_gemm(
                input_tensor_quant,
                input_tensor_scale,
                self.weight,
                self.weight_scale.t().float(),
                output_dtype=self.infer_dtype,
            )
        else:
            # Fallback: dequantize and use standard matmul
            dtype = input_tensor.dtype

            # Ensure weight and scales are on the same device
            weight = self.weight.to(input_tensor.device)
            weight_scale = self.weight_scale.to(input_tensor.device)

            # Dequantize weight: weight_fp32 = weight_int8 * weight_scale
            weight_fp32 = weight.to(torch.float32) * weight_scale.squeeze(-1)

            # Dequantize activation: input_fp32 = input_int8 * input_scale
            input_fp32 = input_tensor_quant.to(torch.float32) * input_tensor_scale

            # Perform matrix multiplication
            output_tensor = torch.matmul(input_fp32, weight_fp32)

            # Convert back to original dtype
            output_tensor = output_tensor.to(dtype)

        # Add bias if present
        if self.bias is not None:
            bias = self.bias.to(input_tensor.device)
            output_tensor = output_tensor + bias

        return output_tensor

