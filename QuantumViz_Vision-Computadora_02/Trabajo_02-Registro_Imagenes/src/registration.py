"""
registration.py

Mejoras y utilidades para el registro (stitching) de tres imágenes:

- Detección/descrición (usa funciones externas).
- Emparejamiento con ratio test (usa función externa de matching).
- Estimación de homografías respecto a la imagen 1 (referencia).
- Warp de img2 y img3 al marco de img1.
- Blending por feathering (alpha blending ponderado por distancia).

Notas
-----
Este módulo asume que existen dos funciones externas:
- detector_fn(img_gray, method=...) -> keypoints, descriptors
- matcher_fn(descriptors_query, descriptors_train, k=2) -> knn matches

La función ratio_test_filter debe devolver una lista de cv2.DMatch "buenos".
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import cv2


# --------------------------
# Utilidades para homografías
# --------------------------
def find_homography_ransac(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_thresh: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estima homografía con RANSAC entre kp2 (source) -> kp1 (dst) a partir de
    matches.

    Parameters
    ----------
    kp1 : list[cv2.KeyPoint]
        Keypoints de la imagen destino (img1).
    kp2 : list[cv2.KeyPoint]
        Keypoints de la imagen fuente (img2 o img3).
    matches : list[cv2.DMatch]
        Matches filtrados (por ejemplo por ratio test), donde:
        - queryIdx corresponde a kp1 (dest)
        - trainIdx corresponde a kp2 (src)
    ransac_thresh : float
        Umbral de reproyección para RANSAC.

    Returns
    -------
    H : np.ndarray or None
        Matriz de homografía 3x3 si se estima correctamente, o None si no hay
        suficientes matches.
    mask : np.ndarray or None
        Máscara de inliers devuelta por cv2.findHomography.
    """
    if matches is None or len(matches) < 4:
        return None, None

    try:
        src_pts = np.array(
            [kp2[m.trainIdx].pt for m in matches], dtype=np.float32
        ).reshape((-1, 1, 2))
        dst_pts = np.array(
            [kp1[m.queryIdx].pt for m in matches], dtype=np.float32
        ).reshape((-1, 1, 2))
    except Exception as e:
        # Manejo defensivo si los matches no son DMatch
        raise ValueError("Formato de matches inesperado en find_homography_ransac") from e

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    return H, mask


# --------------------------
# Canvas / Warping
# --------------------------
def create_canvas_for_warp(
    img_refs: List[np.ndarray], Hs: List[np.ndarray]
) -> Tuple[int, int, int, int]:
    """
    Calcula el tamaño del canvas que contendrá la imagen de referencia y las imágenes warpeadas.

    Parameters
    ----------
    img_refs : list[np.ndarray]
        Lista de imágenes (BGR) donde img_refs[0] es la referencia.
    Hs : list[np.ndarray]
        Lista de homografías que mapean cada imagen i al sistema de coordenadas de img_refs[0].
        El primer elemento normalmente es la identidad para img_refs[0].

    Returns
    -------
    width, height, offset_x, offset_y : tuple[int]
        Tamaño del canvas y offset (en píxeles) para trasladar coordenadas positivas.
    """
    # Obtener los corner points de la imagen de referencia
    h0, w0 = img_refs[0].shape[:2]
    corners0 = np.array(
        [
            [0, 0],
            [0, h0],
            [w0, h0],
            [w0, 0]
        ],
        dtype=np.float32
    ).reshape(-1, 1, 2)
    all_pts = corners0.copy()

    # Transformar los corners de cada imagen al frame de la referencia
    for i in range(1, len(img_refs)):
        h, w = img_refs[i].shape[:2]
        corners = np.array(
            [
                [0, 0], [0, h], [w, h], [w, 0]
            ],
            dtype=np.float32
        ).reshape(-1, 1, 2)
        # Si la homografía no está dada, se asume identidad (no mover)
        H = Hs[i] if Hs[i] is not None else np.eye(3)
        warped = cv2.perspectiveTransform(corners, H)
        all_pts = np.concatenate((all_pts, warped), axis=0)

    mins = (all_pts.min(axis=0).ravel() - 0.5).astype(np.int32)
    xmin, ymin = mins[0], mins[1]
    maxs = (all_pts.max(axis=0).ravel() + 0.5).astype(np.int32)
    xmax, ymax = maxs[0], maxs[1]

    width = int(xmax - xmin)
    height = int(ymax - ymin)
    offset_x = int(-xmin)
    offset_y = int(-ymin)
    return width, height, offset_x, offset_y


def warp_image_to_canvas(
    img: np.ndarray, H: np.ndarray, canvas_size: Tuple[int, int], offset: Tuple[int, int]
) -> np.ndarray:
    """
    Warp de una imagen al canvas usando homografía H y offset.

    Parameters
    ----------
    img : np.ndarray
        Imagen fuente a warpear.
    H : np.ndarray
        Homografía que mapea img -> frame de referencia (img1).
    canvas_size : (width, height)
        Tamaño del canvas destino (como espera cv2.warpPerspective).
    offset : (tx, ty)
        Traducción para trasladar coordenadas al canvas (positiva).

    Returns
    -------
    warped : np.ndarray
        Imagen warpeada con tamaño canvas (BGR).
    """
    tx, ty = offset
    H_trans = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_full = H_trans.dot(H if H is not None else np.eye(3))
    warped = cv2.warpPerspective(
        img, H_full, canvas_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return warped


# --------------------------
# Blending (feathering)
# --------------------------
def feather_blend(
    canvas: np.ndarray, images: List[np.ndarray], masks: List[np.ndarray]
) -> np.ndarray:
    """
    Realiza blending con feathering (alpha según distancia interna de la máscara).

    Parameters
    ----------
    canvas : np.ndarray
        Canvas base (normalmente contiene la img1).
    images : list[np.ndarray]
        Lista de imágenes alineadas con el canvas (mismo shape).
    masks : list[np.ndarray]
        Lista de máscaras binarias (uint8 con 0/1) indicando región válida de cada imagen.

    Returns
    -------
    blended_final : np.ndarray
        Imagen resultante blendada (uint8, BGR).
    """
    eps = 1e-8
    h, w = masks[0].shape
    alpha_maps: List[np.ndarray] = []

    # Construir mapas alpha basados en la distance transform para cada máscara
    for m in masks:
        if m.dtype != np.uint8:
            m_bin = (m > 0).astype(np.uint8)
        else:
            m_bin = (m > 0).astype(np.uint8)

        # distanceTransform necesita 0 para background y 255 para foreground
        dist = cv2.distanceTransform((m_bin * 255).astype(np.uint8), cv2.DIST_L2, 5)
        if dist.max() > 0:
            a = dist / (dist.max() + eps)
        else:
            a = dist.astype(np.float32)
        alpha_maps.append(a)

    alpha_sum = np.sum(alpha_maps, axis=0) + eps
    alphas = [a / alpha_sum for a in alpha_maps]

    blended = np.zeros_like(images[0], dtype=np.float32)
    for img, a in zip(images, alphas):
        A = np.dstack([a, a, a])  # expand alpha a 3 canales
        blended += img.astype(np.float32) * A

    # Donde no contribuye ninguna imagen, conservar el canvas original
    mask_any = np.clip(np.sum(masks, axis=0), 0, 1).astype(bool)
    canvas_rgb = canvas.astype(np.float32)
    blended_final = blended.copy()
    blended_final[~mask_any] = canvas_rgb[~mask_any]

    return np.clip(blended_final, 0, 255).astype(np.uint8)


# --------------------------
# Pipeline principal
# --------------------------
def stitch_three_images(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    detector_fn: Callable[[np.ndarray, str], Tuple[List[cv2.KeyPoint], np.ndarray]],
    matcher_fn: Callable[..., List],
    method: str = "SIFT",
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]]]:
    """
    Registra tres imágenes en el marco de img1 (referencia) y realiza blending.

    Parameters
    ----------
    img1, img2, img3 : np.ndarray
        Imágenes BGR.
    detector_fn : callable
        Función detector/descriptor: detector_fn(gray_img, method=...) -> (keypoints, descriptors)
    matcher_fn : callable
        Función para realizar knn match: matcher_fn(des_query, des_train, k=2) -> knn matches.
        Se asume que existe una función externa ratio_test_filter() que toma esos knn matches y
        devuelve una lista de cv2.DMatch "buenos".
    method : str
        Método a pasar a detector_fn (ej. "SIFT", "ORB").
    ratio : float
        Ratio para el filtro de Lowe (se pasa a ratio_test_filter).
    ransac_thresh : float
        Umbral para cv2.findHomography RANSAC.

    Returns
    -------
    pano : np.ndarray
        Imagen pano blendada en el frame de img1.
    homographies : dict
        Diccionario con claves 'H12', 'H13' y 'offset' para referencia.
    """
    # Convertir a gris para detector
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    # Helpers resistentes a distintas firmas de detector/matcher
    def _call_detector(det_fn, gray_img, method_val):
        try:
            return det_fn(gray_img, method=method_val)
        except TypeError:
            # intentar sin el argumento 'method'
            return det_fn(gray_img)

    def _call_matcher(match_fn, des_query, des_train, k_val=2):
        try:
            return match_fn(des_query, des_train, k=k_val)
        except TypeError:
            # intentar sin el argumento 'k'
            return match_fn(des_query, des_train)

    kp1, des1 = _call_detector(detector_fn, gray1, method)
    kp2, des2 = _call_detector(detector_fn, gray2, method)
    kp3, des3 = _call_detector(detector_fn, gray3, method)

    if des1 is None or des2 is None or des3 is None:
        raise RuntimeError("No se encontraron descriptores en una de las imágenes.")

    # knn matching 1<->2 y 1<->3 (query: des1)
    matches12_knn = _call_matcher(matcher_fn, des1, des2, k_val=2)
    matches13_knn = _call_matcher(matcher_fn, des1, des3, k_val=2)

    # Filtro por ratio test: utilizamos la función externa esperada
    try:
        from src.matching import ratio_test_filter
    except Exception:
        # Si la importación falla, intentamos búsqueda relativa (más tolerante)
        try:
            from matching import ratio_test_filter  # type: ignore
        except Exception as e:
            raise ImportError(
                "No se pudo importar ratio_test_filter desde 'src.matching' ni 'matching'."
            ) from e

    good12 = ratio_test_filter(matches12_knn, ratio=ratio)
    good13 = ratio_test_filter(matches13_knn, ratio=ratio)

    if len(good12) < 4 and len(good13) < 4:
        raise RuntimeError("Pocos matches buenos para estimar homografías.")

    # Estimar homografías img2->img1 y img3->img1
    H12, mask12 = (
        find_homography_ransac(kp1, kp2, good12, ransac_thresh=ransac_thresh)
        if len(good12) >= 4
        else (None, None)
    )
    H13, mask13 = (
        find_homography_ransac(kp1, kp3, good13, ransac_thresh=ransac_thresh)
        if len(good13) >= 4
        else (None, None)
    )

    # Preparar lista de homografías (indice 0: identidad para img1)
    Hs = [np.eye(3), H12 if H12 is not None else np.eye(3), H13 if H13 is not None else np.eye(3)]

    # Calcular canvas que contenga todo
    canvas_w, canvas_h, offx, offy = create_canvas_for_warp([img1, img2, img3], Hs)
    canvas_size = (canvas_w, canvas_h)

    # Colocar img1 en el canvas (con offset)
    tx, ty = offx, offy
    H_identity = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    canvas = cv2.warpPerspective(img1, H_identity, canvas_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Warp img2 e img3 al canvas
    warped2 = warp_image_to_canvas(img2, Hs[1], canvas_size, (offx, offy))
    warped3 = warp_image_to_canvas(img3, Hs[2], canvas_size, (offx, offy))

    # Máscaras binarias indicando pixeles válidos
    mask1 = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0).astype("uint8")
    mask2 = (cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 0).astype("uint8")
    mask3 = (cv2.cvtColor(warped3, cv2.COLOR_BGR2GRAY) > 0).astype("uint8")

    # Feather blending
    blended = feather_blend(canvas, [canvas, warped2, warped3], [mask1, mask2, mask3])

    homographies = {"H12": H12, "H13": H13, "offset": (offx, offy)}
    return blended, homographies


if __name__ == "__main__":
    print("registration module improved for 3-image stitching")
