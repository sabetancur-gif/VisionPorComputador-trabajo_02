"""measurement.py

Herramientas para calibrar escala usando objetos de referencia y medir
distancias en la imagen fusionada. Incluye funciones no interactivas y una
función simple basada en eventos de matplotlib para medir distancias en
píxeles.
"""

from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_scale_from_reference(
    pixel_length: float, real_length_m: float
) -> float:
    """Calcula escala (metros por pixel).
    Args:
        pixel_length: longitud medida en píxeles en la imagen.
        real_length_m: longitud real del objeto en metros.
    Returns:
        meters_per_pixel (float)
    """
    if pixel_length <= 0:
        raise ValueError('pixel_length debe ser mayor que cero.')
    return real_length_m / pixel_length


def pixels_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula la distancia euclidiana entre dos puntos en píxeles."""
    x1, y1 = p1
    x2, y2 = p2
    return np.hypot(x2-x1, y2-y1)


def measure_in_meters(p1, p2, meters_per_pixel: float) -> float:
    """Mide la distancia entre dos puntos dados en píxeles y convierte a
    metros."""
    return pixels_distance(p1, p2) * meters_per_pixel


def interactive_measurement(image, meters_per_pixel):
    """Herramienta interactiva (matplotlib) para seleccionar dos puntos y
    medir. Devuelve la distancia en metros.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pts = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        pts.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro')
        fig.canvas.draw()
        if len(pts) == 2:
            dist_m = measure_in_meters(pts[0], pts[1], meters_per_pixel)
            ax.set_title(f'Distancia: {dist_m:.3f} m')
            fig.canvas.draw()
            print(f'Distancia (m): {dist_m:.3f}')
            fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    print('Módulo measurement: calibración y medición en metros.')
