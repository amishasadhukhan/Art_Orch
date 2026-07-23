from .art_orch_bridge import ArtOrchBridgeExtension
from krita import Krita

Krita.instance().addExtension(ArtOrchBridgeExtension(Krita.instance()))
