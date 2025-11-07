"""
utils.py

Módulo de utilidades para operaciones comunes en procesamiento de imágenes:
- Lectura y validación de imágenes desde disco.
- Visualización de puntos clave (keypoints) detectados.
- Cálculo del error cuadrático medio (RMSE) entre puntos de referencia
  y puntos estimados, útil para validación sintética.
"""

from typing import List

import cv2  # type: ignore
import numpy as np


def read_img(path: str) -> np.ndarray:
    """
    Lee una imagen desde la ruta especificada.

    Parameters
    ----------
    path : str
        Ruta al archivo de imagen.

    Returns
    -------
    np.ndarray
        Imagen leída en formato BGR (como devuelve OpenCV).

    Raises
    ------
    FileNotFoundError
        Si la imagen no existe o no puede ser leída.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Imagen no encontrada: {path}')
    return img


def draw_keypoints(
        img: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        max_kpts: int = 200
) -> np.ndarray:
    """
    Dibuja los keypoints sobre una imagen para visualización.

    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (en formato BGR o RGB).
    keypoints : list of cv2.KeyPoint
        Lista de keypoints detectados.
    max_kpts : int, optional
        Número máximo de keypoints a dibujar (por defecto 200).

    Returns
    -------
    np.ndarray
        Imagen con los keypoints dibujados.

    Notes
    -----
    Se usa la bandera DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    para mostrar tanto posición como tamaño y orientación.
    """
    k = keypoints[:max_kpts]
    img_k = cv2.drawKeypoints(
        img,
        k,
        img.copy(),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_k


def compute_rmse_transform(gt_pts: np.ndarray, est_pts: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio (RMSE) entre puntos ground truth y estimados.

    Parameters
    ----------
    gt_pts : np.ndarray
        Puntos de referencia (ground truth), de forma (N, 2).
    est_pts : np.ndarray
        Puntos estimados, de forma (N, 2).

    Returns
    -------
    float
        Valor del RMSE.

    Raises
    ------
    ValueError
        Si los arrays no tienen la misma forma.
    """
    if gt_pts.shape != est_pts.shape:
        raise ValueError('gt_pts y est_pts deben tener la misma forma.')

    diff = gt_pts - est_pts
    rmse = np.sqrt((diff ** 2).mean())
    return rmse


if __name__ == '__main__':
    # Punto de entrada opcional para pruebas rápidas o depuración.
    print('Módulo utils: utilidades para IO y métricas.')
