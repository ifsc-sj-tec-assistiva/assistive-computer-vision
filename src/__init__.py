from .face.detector_rosto import DetectorRosto

try:
    from .hand.detector_maos import DetectorMaos
    __all__ = ["DetectorRosto", "DetectorMaos"]
except ImportError:
    __all__ = ["DetectorRosto"]
