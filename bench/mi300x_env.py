"""MI300X environment variable setup from AMD GPU MODE hackathon lessons.

Set these BEFORE importing pyquantum or any HIP code.
Each is zero-risk and provides free performance:
- HIP_FORCE_DEV_KERNARG=1: ~0.5-1us savings per kernel launch
- AMD_DIRECT_DISPATCH=1: ~0.3-0.5us savings per kernel launch
- GPU_MAX_HW_QUEUES=8: more hardware queues for concurrent kernels
- HSA_ENABLE_SDMA=0: disable system DMA engine (can add latency)
"""
import os

os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["AMD_DIRECT_DISPATCH"] = "1"
os.environ["GPU_MAX_HW_QUEUES"] = "8"
os.environ["HSA_ENABLE_SDMA"] = "0"
