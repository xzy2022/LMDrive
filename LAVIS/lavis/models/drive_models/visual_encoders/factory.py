from typing import List


_MEMFUSER_MODEL_NAMES = (
    "memfuser_baseline",
    "memfuser_baseline_return_feature",
    "memfuser_baseline_e3d3",
    "memfuser_baseline_e1d3",
    "memfuser_baseline_e1d3_return_feature",
    "memfuser_baseline_e1d3_r26",
    "memfuser_baseline_e1d3_r26_return_feature",
    "memfuser_baseline_e2d2",
)


def list_visual_encoder_names() -> List[str]:
    return list(_MEMFUSER_MODEL_NAMES)


def create_visual_encoder(model_name: str, **kwargs):
    if model_name not in _MEMFUSER_MODEL_NAMES:
        raise RuntimeError(
            "Unknown visual encoder ({}) in LMDrive local factory. Supported models: {}"
            .format(model_name, ", ".join(list_visual_encoder_names()))
        )

    # Import lazily so the rest of LMDrive can still import even when the host
    # torch_scatter binary is not usable. The error is raised when the encoder
    # is actually instantiated or executed.
    from .memfuser import create_memfuser

    return create_memfuser(model_name, **kwargs)
