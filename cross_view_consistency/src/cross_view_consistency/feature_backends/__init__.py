"""Feature backend registry."""

from .base import FeatureBackend, FeatureResult
from .dinov2_backend import PublicDinoV2Backend
from .lingbot_backbone_backend import LingbotBackbonePreAggBackend
from .lingbot_single_frame_backend import LingbotSingleFrameReconBackend
from .lingbot_offline_global_backend import LingbotOfflineGlobalBackend


def build_backend(name: str) -> FeatureBackend:
    registry = {
        "public_dinov2_vitl14": PublicDinoV2Backend,
        "lingbot_backbone_pre_agg_rope_off": LingbotBackbonePreAggBackend,
        "lingbot_single_frame_recon_rope_off": LingbotSingleFrameReconBackend,
        "lingbot_offline_global_32f_rope_off": LingbotOfflineGlobalBackend,
    }
    if name not in registry:
        raise KeyError(f"Unknown feature backend: {name}")
    return registry[name]()

