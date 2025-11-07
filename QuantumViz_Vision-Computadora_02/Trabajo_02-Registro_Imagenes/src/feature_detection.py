"""feature_detection.py

Funciones para detección de puntos clave y extracción de descriptores.
Incluye SIFT y ORB (si están disponibles).
Cada función devuelve keypoints y descriptores compatibles con OpenCV.
"""

from typing import Tuple, List
import cv2
import numpy as np


def sift_detector(nfeatures: int = 0):
    """Crear un detector SIFT. Requiere opencv-contrib-python."""
    sift_ctor = getattr(cv2, "SIFT_create", None)
    if sift_ctor is None:
        xfs = getattr(cv2, "xfeatures2d", None)
        sift_ctor = getattr(xfs, "SIFT_create", None) if xfs is not None else None
    if sift_ctor is None:
        raise RuntimeError("SIFT no está disponible en esta instalación de OpenCV. Instale opencv-contrib-python.")
    try:
        # Algunas versiones aceptan el argumento nfeatures; otras no.
        return sift_ctor(nfeatures=nfeatures)  # type: ignore[call-arg, attr-defined]
    except TypeError:
        return sift_ctor()  # type: ignore[call-arg, attr-defined]


def orb_detector(nfeatures: int = 5000):
    """Crear un detector ORB (alternativa rápida).
    Usa getattr para evitar advertencias del linter si los stubs no exponen ORB_create.
    """
    orb_ctor = getattr(cv2, "ORB_create", None)
    # raramente ORB está en xfeatures2d, pero se comprueba por si acaso
    if orb_ctor is None:
        xfs = getattr(cv2, "xfeatures2d", None)
        orb_ctor = getattr(xfs, "ORB_create", None) if xfs is not None else None
    if orb_ctor is None:
        raise RuntimeError("ORB no está disponible en esta instalación de OpenCV. Instale opencv-python.")
    try:
        return orb_ctor(nfeatures=nfeatures)  # type: ignore[call-arg, attr-defined]
    except TypeError:
        return orb_ctor()  # type: ignore[call-arg, attr-defined]


def detect_and_describe(img: np.ndarray, method: str = 'SIFT') -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detectar keypoints y calcular descriptores en una imagen.
    Args:
        img: imagen en escala de grises (np.ndarray).
        method: 'SIFT' o 'ORB'.
    Returns:
        keypoints, descriptors
    """
    if method.upper() == 'SIFT':
        detector = sift_detector()
    elif method.upper() == 'ORB':
        detector = orb_detector()
    else:
        raise ValueError('Método desconocido: use SIFT u ORB')

    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


if __name__ == '__main__':
    print('Este módulo exporta funciones para detección de características.')
