"""Make the project root importable when a script is run as `python scripts/<name>.py`."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# expandable segments (less fragmentation): torch 2.9.1's CUDA allocator still reads them only from the deprecated
# PYTORCH_CUDA_ALLOC_CONF (PYTORCH_ALLOC_CONF alone has no effect: is_expandable is False in torch.cuda.memory_snapshot);
# with both set they are enabled without the deprecation warning
for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
    os.environ.setdefault(_var, "expandable_segments:True")
# NCCL peer-to-peer across the CPU sockets. The 4 A100 PCIe of this machine are two NVLink pairs (0-1, 2-3) on two
# sockets: by default NCCL routes the cross-socket hops of a 3-4 GPU ring through host shared memory, and an
# all-reduce runs at ~0.6 GB/s (a 1 GB fp32 gradient takes 2.4 s, longer than the whole training step); with P2P
# over PCIe it runs at ~14 GB/s (0.11 s). This alone made the OpenWebText trainings ~2-4x slower than the GPUs allow.
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
