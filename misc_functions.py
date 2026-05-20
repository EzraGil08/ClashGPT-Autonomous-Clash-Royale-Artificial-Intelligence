import numpy as np
import cv2 as cv

ELIXIR_BLOB_FILE = r"C:\Users\Ezra\OneDrive\Desktop\Senior Research Project\Images\elixir_blob.png"
ELIXIR_BLOB = cv.imread(ELIXIR_BLOB_FILE, 0)

def get_top_corner(img):
    bgr_img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

    # Target in BGR (reversed from RGB 24,30,72)
    target    = np.array([72, 30, 24], dtype=np.int32)
    tolerance = 5

    h, w = bgr_img.shape[:2]

    for y in range(h):
        for x in range(w):
            pixel = bgr_img[y, x].astype(np.int32)
            if (np.abs(pixel - target) <= tolerance).all():
                return (x, y)

    return None


def get_elixir(img, corner, last_elixir: float = 0.0) -> float:
    """
    Read the elixir bar from the game frame.

    Parameters
    ----------
    img          : BGR or BGRA frame from mss/cv2.
    corner       : (x, y) top-left of the game window, from get_top_corner().
    last_elixir  : elixir value from the previous step. Used to suppress
                   visual spikes — elixir can only rise slowly, so any reading
                   more than 0.15 above last_elixir is clamped. Pass 0.0 on
                   the first call (no clamping applied when last is 0).

    Returns
    -------
    float in [0.0, 10.0], rounded to 2 decimal places.
    """
    bgr_img          = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    h, w             = bgr_img.shape[:2]
    lower_elixir     = np.array([143, 190, 150])
    upper_elixir     = np.array([153, 240, 230])
    lower_light_blue = np.array([104, 150, 119])
    upper_light_blue = np.array([117, 218, 175])

    # Early-exit bounds check
    check_x = 538 + corner[0]
    check_y = 978 + corner[1]
    if check_x >= w or check_y >= h:
        return last_elixir

    # Early-exit: check if bar is full
    bgr_check = bgr_img[check_y, check_x].reshape(1, 1, 3)
    hsv_check = cv.cvtColor(bgr_check, cv.COLOR_BGR2HSV)[0][0]
    if (lower_elixir <= hsv_check).all() and (hsv_check <= upper_elixir).all():
        raw = 10.0
        # Still clamp — a jump from 3 to 10 in one frame is a spike
        if last_elixir > 0.0 and raw > last_elixir + 0.15:
            return round(last_elixir + 0.15, 2)
        return 10.0

    x_bars = [154, 194, 233, 272, 311, 350, 389, 428, 468, 507]
    y_bar  = 978

    # Bounds check for bar scan
    if any(x + corner[0] >= w or y_bar + corner[1] >= h for x in x_bars):
        return last_elixir

    elixir = 0.0

    for x_bar in x_bars:
        px = x_bar + corner[0]
        py = y_bar + corner[1]
        bgr_pixel = bgr_img[py, px].reshape(1, 1, 3)
        hsv_pixel = cv.cvtColor(bgr_pixel, cv.COLOR_BGR2HSV)[0][0]

        if (lower_elixir <= hsv_pixel).all() and (hsv_pixel <= upper_elixir).all():
            elixir += 1
        else:
            for var in range(39):
                px2 = x_bar + corner[0] + var
                if px2 >= w:
                    break
                bgr_pixel = bgr_img[py, px2].reshape(1, 1, 3)
                hsv_pixel = cv.cvtColor(bgr_pixel, cv.COLOR_BGR2HSV)[0][0]
                if (lower_light_blue <= hsv_pixel).all() and (hsv_pixel <= upper_light_blue).all():
                    elixir += 1 / 39
                else:
                    break
            break

    raw = round(elixir, 2)

    # Spike suppression: elixir rises at ~0.3/sec; more than 0.15 above last
    # reading in a single step is a visual artifact. Only clamp upward spikes —
    # downward drops (card played) are always valid.
    # Skip clamping if last_elixir is 0 (first call / unknown state).
    if raw > last_elixir + 0.15:
        return round(last_elixir + 0.15, 2)

    return raw


# ── TEST ─────────────────────────────────────────────────────
# GAME_FILE = r"C:\Users\Ezra\OneDrive\Desktop\Senior Research Project\Images\test_images\GameT.png"
# GAME = cv.imread(GAME_FILE)
# print(get_top_corner(GAME))
# print(get_elixir(GAME, get_top_corner(GAME)))