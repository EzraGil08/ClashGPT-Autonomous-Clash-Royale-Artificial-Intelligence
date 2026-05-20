import cv2 as cv
import numpy as np

# ── Anchor reference ───────────────────────────────────────────────────────────

ELIXIR_BLOB_REF = (144, 969)

BARS = {
    "ally_left":   (113, 172, 618),
    "ally_right":  (407, 466, 618),
    "enemy_left":  (113, 172, 151),
    "enemy_right": (407, 466, 151),
}

KING_BARS = {
    "ally_king":  (251, 332, 754),
    "enemy_king": (250, 333, 39),
}

# Which princess towers gate each king bar
KING_PRINCESS_PAIRS = {
    "ally_king":  ("ally_left",  "ally_right"),
    "enemy_king": ("enemy_left", "enemy_right"),
}

# ── Colors (BGR) ──────────────────────────────────────────────────────────────

ALLY_HEALTH_COLORS = np.array([
    [205, 164, 97],
    [255, 210, 114],
    [250, 208, 114],
], dtype=np.uint8)

ALLY_TERMINATE_COLOR = np.array([[113, 80, 65]], dtype=np.uint8)

ENEMY_HEALTH_COLORS = np.array([
    [80, 38, 204],
    [94, 38, 226],
    [84, 31, 207],
    [80, 46, 140]
], dtype=np.uint8)

ENEMY_TERMINATE_COLORS = np.array([
    [73, 50, 93],
    [74, 50, 98],
], dtype=np.uint8)

ALLY_KING_HEALTH_COLORS = np.array([
    [255, 210, 114],   # RGB(114,210,255) → BGR(255,210,114)
], dtype=np.uint8)

ENEMY_KING_HEALTH_COLORS = np.array([
    [93, 37, 222],     # RGB(222,37,93) → BGR(93,37,222)
], dtype=np.uint8)

COLOR_TOLERANCE = 18

# How many consecutive frames the king bar must show health before a princess
# tower is officially declared dead. Guards against a single noisy frame
# triggering a premature death declaration on episode start.
KING_CONFIRM_FRAMES = 3

ALL_TOWER_KEYS = list(BARS.keys()) + list(KING_BARS.keys())


# ── State factory ──────────────────────────────────────────────────────────────

def make_tower_state() -> dict:
    """
    Returns a fresh per-episode tower state dict. Pass this to every
    get_tower_health() call; it is mutated in-place.

    Fields
    ------
    dead           : set[str]  — towers permanently locked at 0
    king_confirm   : dict[str, int]  — consecutive frames king bar has shown health
                     (used to confirm a princess is dead before locking it)
    """
    return {
        "dead":         set(),
        "king_confirm": {k: 0 for k in KING_BARS},
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _matches_any(pixel, palette, tol):
    diff = np.abs(palette.astype(int) - pixel.astype(int))
    return bool(np.any(np.all(diff <= tol, axis=1)))


def _scan_princess(image, x_start, x_end, y, health_colors, terminate_colors, w, h):
    """
    Scan a princess tower bar. Returns (pct, xs, hit_unexpected).
    hit_unexpected=True means the first pixel was neither health nor terminate.
    """
    bar_length = x_end - x_start + 1
    health_pixels = 0
    xs = []

    for x in range(x_start, x_end + 1):
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        pixel = image[y, x]
        if _matches_any(pixel, health_colors, COLOR_TOLERANCE):
            health_pixels += 1
            xs.append(x)
        elif _matches_any(pixel, terminate_colors, COLOR_TOLERANCE):
            break
        else:
            return None, [], True   # unexpected — caller will check king bar

    return round((health_pixels / bar_length) * 100, 1), xs, False


def _scan_king(image, x_start, x_end, y, health_colors, w, h):
    """
    Scan a king tower bar (stop-on-first-unknown).
    Returns (pct, xs). pct=0 means the bar showed no health pixels.
    """
    bar_length = x_end - x_start + 1
    health_pixels = 0
    xs = []

    for x in range(x_start, x_end + 1):
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        pixel = image[y, x]
        if _matches_any(pixel, health_colors, COLOR_TOLERANCE):
            health_pixels += 1
            xs.append(x)
        else:
            break   # first non-health pixel = end of bar

    return round((health_pixels / bar_length) * 100, 1), xs


# ── Core ───────────────────────────────────────────────────────────────────────

def get_tower_health(image, corner, last_known_health, tower_state):
    """
    Parameters
    ----------
    image             : BGR screenshot
    corner            : (dx, dy) from get_top_corner()
    last_known_health : dict[str, float] — previous frame's health values
    tower_state       : dict from make_tower_state(), mutated in-place

    Returns
    -------
    results         : dict[str, float] — health % for all 6 towers
    health_segments : dict[str, list[int]] — x positions of health pixels
    """
    dx, dy = corner
    h, w = image.shape[:2]

    results         = {}
    health_segments = {}
    dead            = tower_state["dead"]
    king_confirm    = tower_state["king_confirm"]

    # ── Princess towers ────────────────────────────────────────────────────────
    for tower, (x_start_ref, x_end_ref, y_ref) in BARS.items():

        # Already confirmed dead — skip scan
        if tower in dead:
            results[tower]         = 0.0
            health_segments[tower] = []
            continue

        is_ally = tower.startswith("ally")
        x_start = x_start_ref + dx
        x_end   = x_end_ref   + dx
        y       = y_ref       + dy

        health_colors    = ALLY_HEALTH_COLORS   if is_ally else ENEMY_HEALTH_COLORS
        terminate_colors = ALLY_TERMINATE_COLOR if is_ally else ENEMY_TERMINATE_COLORS

        pct, xs, hit_unexpected = _scan_princess(
            image, x_start, x_end, y, health_colors, terminate_colors, w, h
        )

        if hit_unexpected:
            # Unexpected color on princess bar — peek at the king bar to confirm
            king_key = "ally_king" if is_ally else "enemy_king"
            kx0, kx1, ky = KING_BARS[king_key]
            king_colors  = ALLY_KING_HEALTH_COLORS if is_ally else ENEMY_KING_HEALTH_COLORS

            king_pct, _ = _scan_king(
                image,
                kx0 + dx, kx1 + dx, ky + dy,
                king_colors, w, h,
            )

            if king_pct > 0:
                # King bar is showing health → this princess is dead
                king_confirm[king_key] += 1
                if king_confirm[king_key] >= KING_CONFIRM_FRAMES:
                    dead.add(tower)
                    print(f"[tower_health] {tower} confirmed DEAD "
                          f"(king bar active for {KING_CONFIRM_FRAMES} frames)")
                    pct = 0.0
                else:
                    # Not enough confirmation frames yet — hold last-known
                    pct = last_known_health.get(tower, 0)
            else:
                # King bar empty — animation still playing, hold last-known
                king_confirm[king_key] = 0
                pct = last_known_health.get(tower, 0)

            xs = []

        results[tower]         = pct
        health_segments[tower] = xs

    # ── King towers ────────────────────────────────────────────────────────────
    # Always scan king bars whose side has at least one confirmed-dead or
    # currently-0 princess, so we keep the HP reading current.
    for king_key, (x_start_ref, x_end_ref, y_ref) in KING_BARS.items():
        p1, p2 = KING_PRINCESS_PAIRS[king_key]

        side_princess_down = results[p1] == 0 or results[p2] == 0
        if not side_princess_down:
            # No princess down on this side — king not yet active
            results[king_key]         = last_known_health.get(king_key, 100)
            health_segments[king_key] = []
            king_confirm[king_key]    = 0
            continue

        if king_key in dead:
            results[king_key]         = 0.0
            health_segments[king_key] = []
            continue

        is_ally     = king_key == "ally_king"
        king_colors = ALLY_KING_HEALTH_COLORS if is_ally else ENEMY_KING_HEALTH_COLORS

        pct, xs = _scan_king(
            image,
            x_start_ref + dx, x_end_ref + dx, y_ref + dy,
            king_colors, w, h,
        )

        # If king bar reads 0 for KING_CONFIRM_FRAMES consecutive frames,
        # declare it dead too.
        if pct == 0:
            king_confirm[king_key] += 1
            if king_confirm[king_key] >= KING_CONFIRM_FRAMES:
                dead.add(king_key)
                print(f"[tower_health] {king_key} confirmed DEAD")
                pct = 0.0
            else:
                pct = last_known_health.get(king_key, 100)
            xs = []
        else:
            king_confirm[king_key] = 0

        results[king_key]         = pct
        health_segments[king_key] = xs

    return results, health_segments


# ── Debug visualizer ──────────────────────────────────────────────────────────

def debug_draw(image, corner, last_known_health, tower_state=None,
               out_path="tower_health_debug.png"):
    """
    Runs get_tower_health and draws every scanned pixel onto a copy of the image.

    Pixel colors on output:
      Green  (0,255,0)   — matched health color
      Orange (0,165,255) — bar end (terminate pixel or king bar stop)
      Yellow (0,255,255) — princess unexpected color; king bar being polled
      Dark red line      — tower confirmed DEAD
      Gray   (80,80,80)  — king bar not yet active

    Each bar gets a label with HP% and state annotations.
    """
    if tower_state is None:
        tower_state = make_tower_state()

    dx, dy = corner
    h, w = image.shape[:2]
    out = image.copy()

    state_copy = {
        "dead":         set(tower_state["dead"]),
        "king_confirm": dict(tower_state["king_confirm"]),
    }
    results, _ = get_tower_health(image, corner, last_known_health, state_copy)

    side_down = {
        "ally":  results["ally_left"]  == 0 or results["ally_right"]  == 0,
        "enemy": results["enemy_left"] == 0 or results["enemy_right"] == 0,
    }

    for tower, (x_start_ref, x_end_ref, y_ref) in {**BARS, **KING_BARS}.items():
        is_ally = tower.startswith("ally")
        is_king = tower.endswith("king")
        x_start = x_start_ref + dx
        x_end   = x_end_ref   + dx
        y       = y_ref       + dy

        cv.line(out, (x_start, y), (x_end, y), (40, 40, 40), 1)

        # King inactive
        if is_king and not side_down["ally" if is_ally else "enemy"]:
            cv.line(out, (x_start, y), (x_end, y), (80, 80, 80), 2)
            cv.putText(out, f"{tower}  [inactive]",
                       (x_start, y - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv.LINE_AA)
            continue

        # Confirmed dead
        if tower in tower_state["dead"]:
            cv.line(out, (x_start, y), (x_end, y), (0, 0, 180), 2)
            cv.putText(out, f"{tower}  0.0%  [DEAD]",
                       (x_start, y - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1, cv.LINE_AA)
            continue

        health_colors    = (ALLY_KING_HEALTH_COLORS  if is_king else ALLY_HEALTH_COLORS)    if is_ally else \
                           (ENEMY_KING_HEALTH_COLORS if is_king else ENEMY_HEALTH_COLORS)
        terminate_colors = None if is_king else (ALLY_TERMINATE_COLOR if is_ally else ENEMY_TERMINATE_COLORS)

        confirm_n   = tower_state["king_confirm"].get(tower, 0) if is_king else \
                      tower_state["king_confirm"].get("ally_king" if is_ally else "enemy_king", 0)
        hit_unexpected = False

        for x in range(x_start, x_end + 1):
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            pixel = image[y, x]
            if _matches_any(pixel, health_colors, COLOR_TOLERANCE):
                cv.circle(out, (x, y), 2, (0, 255, 0), -1)
            elif is_king:
                cv.circle(out, (x, y), 2, (0, 165, 255), -1)
                break
            elif _matches_any(pixel, terminate_colors, COLOR_TOLERANCE):
                cv.circle(out, (x, y), 2, (0, 165, 255), -1)
                break
            else:
                cv.circle(out, (x, y), 2, (0, 255, 255), -1)   # yellow = unexpected, king polled
                hit_unexpected = True
                break

        pct         = results[tower]
        state_tag   = f"  [confirming {confirm_n}/{KING_CONFIRM_FRAMES}]" if hit_unexpected else ""
        label       = f"{tower}  {pct}%{state_tag}"
        label_color = (0, 200, 255) if is_ally else (255, 100, 100)
        cv.putText(out, label,
                   (x_start, y - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1, cv.LINE_AA)

    cv.imwrite(out_path, out)
    print(f"[debug_draw] saved → {out_path}")
    return out


# ── TEST ─────────────────────────────────────────────────────
# from misc_functions import get_top_corner

# GAME_FILE = r"C:\Users\Ezra\OneDrive\Desktop\Senior Research Project\Images\test_images\GameK.png"
# GAME = cv.imread(GAME_FILE)

# corner = get_top_corner(GAME)
# last  = {"ally_left": 100, "ally_right": 100, "enemy_left": 100, "enemy_right": 100,
#          "ally_king": 100, "enemy_king": 100}
# state = make_tower_state()
# health, segs = get_tower_health(GAME, corner, last, state)
# for tower, hp in health.items(): print(f"{tower}: {hp}%")

# debug_draw(GAME, corner, last, state, out_path="tower_health_debug.png")