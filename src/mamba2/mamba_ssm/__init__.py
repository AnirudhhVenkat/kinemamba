__version__ = "2.3.2.post1"

# Kinemamba only ships Mamba2 (+ Triton SSD ops it depends on).
from mamba_ssm.modules.mamba2 import Mamba2

__all__ = ["Mamba2"]
