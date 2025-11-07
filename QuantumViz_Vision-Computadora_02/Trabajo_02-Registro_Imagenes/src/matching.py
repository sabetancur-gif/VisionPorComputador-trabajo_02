"""matching.py

Funciones para emparejamiento de descriptores entre dos o tres imágenes.
Incluye BFMatcher y FLANN, y el test de Lowe (ratio test).
"""

from typing import Dict, Any, cast
import numpy as np
import cv2


def flann_match(des1, des2, k=2):
    # FLANN params for SIFT (float descriptors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(
        cast(Dict[str, Any], index_params),
        cast(Dict[str, Any], search_params)
    )
    matches = flann.knnMatch(des1, des2, k=k)
    return matches


def bf_match(des1, des2, crossCheck=False):
    # choose normType based on descriptor dtype
    if des1 is None or des2 is None:
        return []
    if des1.dtype == np.uint8:
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2
    matcher = cv2.BFMatcher(norm, crossCheck=crossCheck)
    matches = matcher.knnMatch(des1, des2, k=2)
    return matches


def ratio_test_filter(matches, ratio=0.75):
    """Aplica el test de Lowe para filtrar matches proporcionados por knnMatch."""
    good = []
    for m in matches:
        if len(m) == 2:
            if m[0].distance < ratio * m[1].distance:
                good.append(m[0])
    return good


def match_descriptors_three(desc1, desc2, desc3, method: str = 'FLANN', ratio: float = 0.75):
    """Empareja descriptores entre tres imágenes.
    Devuelve tripletas aproximadas de coincidencias (indices en cada descriptor).
    Estrategia simple: match 1-2 y 1-3 y conservar keypoints en 1 que tengan matches consistentes.
    """
    if desc1 is None or desc2 is None or desc3 is None:
        return []

    if method.upper() == 'FLANN':
        m12 = flann_match(desc1, desc2)
        m13 = flann_match(desc1, desc3)
    else:
        m12 = bf_match(desc1, desc2)
        m13 = bf_match(desc1, desc3)

    good12 = ratio_test_filter(m12, ratio=ratio)
    good13 = ratio_test_filter(m13, ratio=ratio)

    # Build a mapping from queryIdx -> trainIdx
    map12 = {m.queryIdx: m.trainIdx for m in good12}
    map13 = {m.queryIdx: m.trainIdx for m in good13}

    triplets = []
    for qidx in map12.keys() & map13.keys():
        triplets.append((qidx, map12[qidx], map13[qidx]))
    return triplets


if __name__ == '__main__':
    print('Módulo de matching: funciones para emparejar descriptores (2 o 3 imágenes).')
