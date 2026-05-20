"""
observation.py — builds the flat observation vector fed to MaskablePPO.

Vector layout (202 floats total):
  [0:150]   30 troops × 5 features  (padded with zeros if < 30 troops)
  [150:186] 4 hand slots × 9 one-hots (CARD_NAMES has 9 entries inc. "empty")
  [186:195] next-card one-hot (9)
  [195]     elixir normalised to [0, 1]  (divide by 10)
  [196:200] princess tower healths [ally_left, ally_right, enemy_left, enemy_right]
            normalised to [0, 1]  (divide by 100)
  [200]     ally_king health  normalised to [0, 1]
  [201]     enemy_king health normalised to [0, 1]

Troop feature vector (5 floats per troop):
  0  norm_x        bbox centre x / game_width   [0, 1]
  1  norm_y        bbox centre y / game_height  [0, 1]
  2  is_ally       1.0 = ally, 0.0 = enemy
  3  class_id_norm troop type index / n_classes  [0, 1]
  4  track_age_norm  min(age, MAX_AGE) / MAX_AGE  [0, 1]

Public API
----------
build_observation(tracks, hand, next_card, elixir, tower_health)
    -> np.ndarray shape (202,) dtype float32
"""

import numpy as np
from tracker import TrackResult
from constants import CARD_NAMES

# ── Constants ─────────────────────────────────────────────────────────────────

GAME_W      = 561.0
GAME_H      = 998.0
MAX_TROOPS  = 30
TROOP_FEATS = 5
MAX_AGE     = 60.0

TROOP_CLASSES = [
    # LBGG — player deck units
    "giant", "bowler", "witch", "graveyard", "guard",
    "arrows", "snowball", "minion",
    # LBGG — opponent deck units
    "princess", "ice_spirit", "rocket",
    "goblin_barrel", "knight", "log", "cannon",
    # Spawned units
    "skeleton", "goblin", "spear_goblin",
]
N_TROOP_CLASSES = len(TROOP_CLASSES)
_CLASS_INDEX    = {c: i for i, c in enumerate(TROOP_CLASSES)}

N_CARD_CLASSES = len(CARD_NAMES)          # 9
_CARD_INDEX    = {c: i for i, c in enumerate(CARD_NAMES)}

# Derived sizes
OBS_TROOPS  = MAX_TROOPS * TROOP_FEATS    # 150
OBS_HAND    = 4 * N_CARD_CLASSES          # 36
OBS_NEXT    = N_CARD_CLASSES              # 9
OBS_ELIXIR  = 1
OBS_TOWERS  = 6                           # 4 princess + 2 king
OBS_SIZE    = OBS_TROOPS + OBS_HAND + OBS_NEXT + OBS_ELIXIR + OBS_TOWERS  # 202

# Tower order in the obs vector
TOWER_ORDER = [
    "ally_left", "ally_right", "enemy_left", "enemy_right",
    "ally_king", "enemy_king",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _troop_features(t: TrackResult) -> np.ndarray:
    cx       = ((t.x1 + t.x2) / 2.0) / GAME_W
    cy       = ((t.y1 + t.y2) / 2.0) / GAME_H
    is_ally  = 1.0 if t.team == "ally" else 0.0
    cls_norm = _CLASS_INDEX.get(t.cls, 0) / max(N_TROOP_CLASSES - 1, 1)
    age_norm = min(t.age, MAX_AGE) / MAX_AGE
    return np.array([cx, cy, is_ally, cls_norm, age_norm], dtype=np.float32)


def _card_onehot(card_name: str) -> np.ndarray:
    vec = np.zeros(N_CARD_CLASSES, dtype=np.float32)
    idx = _CARD_INDEX.get(card_name, _CARD_INDEX["empty"])
    vec[idx] = 1.0
    return vec


# ── Public ────────────────────────────────────────────────────────────────────

def build_observation(
    tracks:       list[TrackResult],
    hand:         list[str],
    next_card:    str,
    elixir:       float,
    tower_health: dict[str, float],
) -> np.ndarray:
    """
    Parameters
    ----------
    tracks       : confirmed tracks from CRTracker.update()
    hand         : list[str] length 4, left-to-right (may include "empty")
    next_card    : str — next card label from get_hand()
    elixir       : float in [0, 10]
    tower_health : dict with keys ally_left, ally_right, enemy_left, enemy_right,
                   ally_king, enemy_king — values in [0, 100]

    Returns
    -------
    np.ndarray shape (202,) dtype float32, all values in [0, 1]
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # ── Troop block [0:150] ───────────────────────────────────────────────────
    troops = list(tracks)

    if len(troops) > MAX_TROOPS:
        non_skel = [t for t in troops if t.cls != "skeleton"]
        skel     = [t for t in troops if t.cls == "skeleton"]
        if len(non_skel) >= MAX_TROOPS:
            troops = sorted(non_skel, key=lambda t: t.age)[:MAX_TROOPS]
        else:
            n_skel = MAX_TROOPS - len(non_skel)
            troops = non_skel + sorted(skel, key=lambda t: t.age)[:n_skel]

    for i, t in enumerate(troops[:MAX_TROOPS]):
        start = i * TROOP_FEATS
        obs[start : start + TROOP_FEATS] = _troop_features(t)

    # ── Hand block [150:186] ──────────────────────────────────────────────────
    for slot, card in enumerate(hand[:4]):
        start = OBS_TROOPS + slot * N_CARD_CLASSES
        obs[start : start + N_CARD_CLASSES] = _card_onehot(card)

    # ── Next card [186:195] ───────────────────────────────────────────────────
    obs[OBS_TROOPS + OBS_HAND : OBS_TROOPS + OBS_HAND + OBS_NEXT] = \
        _card_onehot(next_card or "empty")

    # ── Elixir [195] ──────────────────────────────────────────────────────────
    obs[OBS_TROOPS + OBS_HAND + OBS_NEXT] = np.clip(elixir / 10.0, 0.0, 1.0)

    # ── Towers [196:202] ──────────────────────────────────────────────────────
    # Princess towers default to 100 if missing; king towers default to 100
    # (not yet active / not yet seen).
    tower_defaults = {
        "ally_left": 100.0, "ally_right": 100.0,
        "enemy_left": 100.0, "enemy_right": 100.0,
        "ally_king": 100.0, "enemy_king": 100.0,
    }
    base = OBS_TROOPS + OBS_HAND + OBS_NEXT + OBS_ELIXIR
    for j, key in enumerate(TOWER_ORDER):
        val = tower_health.get(key, tower_defaults[key])
        obs[base + j] = np.clip(val / 100.0, 0.0, 1.0)

    return obs


def observation_space_size() -> int:
    """Convenience — returns 202."""
    return OBS_SIZE