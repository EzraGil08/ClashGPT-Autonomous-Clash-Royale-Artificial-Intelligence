"""
constants.py — single source of truth for card data, placement coords,
and action-space layout. Imported by actions.py, the Gym env, and training.py.

All placement coordinates are (x, y) relative to the game-window corner
returned by get_top_corner(). Hand slot coords are also corner-relative.
"""

# ── Card metadata ─────────────────────────────────────────────────────────────

# Ordered list that defines the obs-vector one-hot encoding and hand slots.
CARD_NAMES = [
    "arrows",
    "bowler",
    "giant",
    "graveyard",
    "guards",
    "minions",
    "snowball",
    "witch",
    "empty",          # 9th label used by hand_detection
]

# Cards the player actually runs (no "empty").
PLAYER_DECK = [
    "giant", "bowler", "witch", "graveyard",
    "guards", "arrows", "snowball", "minions",
]

CARD_COSTS: dict[str, int] = {
    "giant":     5,
    "bowler":    5,
    "witch":     5,
    "graveyard": 5,
    "guards":    3,
    "arrows":    3,
    "minions":   3,
    "snowball":  2,
}

# ── Hand slot pixel positions (corner-relative, centre of each card) ──────────
# Slots 0-3 left-to-right; used by execute_action to click the card first.
HAND_SLOT_COORDS: list[tuple[int, int]] = [
    (163, 885),   # slot 0
    (268, 885),   # slot 1
    (373, 885),   # slot 2
    (478, 885),   # slot 3
]

# ── Placement coordinates (corner-relative) ───────────────────────────────────
# Each entry is a list of (x, y) drop points for that card.
CARD_PLACEMENTS: dict[str, list[tuple[int, int]]] = {
    "arrows":    [(80,240),  (480,240), (132,611), (427,610),
                  (132,520), (420,520), (185,302), (373,302)],
    "bowler":    [(56,478),  (135,544), (242,756), (316,760),
                  (426,540), (510,476)],
    "giant":     [(56,478),  (135,544), (242,756), (316,760),
                  (426,540), (510,476)],
    "graveyard": [(80,240),  (480,240)],
    "guards":    [(56,478),  (135,544), (242,756), (316,760),
                  (426,540), (510,476)],
    "minions":   [(56,478),  (135,544), (242,756), (316,760),
                  (426,540), (510,476)],
    "snowball":  [(80,240),  (480,240), (132,611), (427,610),
                  (132,520), (420,520), (185,302), (373,302)],
    "witch":     [(56,478),  (135,544), (242,756), (316,760),
                  (426,540), (510,476)],
}

# ── Action-space layout ───────────────────────────────────────────────────────
# Actions are laid out as contiguous placement blocks per card (deck order),
# followed by a single WAIT action at the end.
#
# ACTION_OFFSETS[card] = index of the first placement action for that card.
# WAIT_ACTION           = index of the wait action.
# N_ACTIONS             = total number of discrete actions.

ACTION_OFFSETS: dict[str, int] = {}
_offset = 0
for _card in PLAYER_DECK:
    ACTION_OFFSETS[_card] = _offset
    _offset += len(CARD_PLACEMENTS[_card])

WAIT_ACTION: int = _offset
N_ACTIONS:   int = _offset + 1   # 49

# Reverse map: action_idx → (card, placement_index) or None for wait
ACTION_INDEX_MAP: dict[int, tuple[str, int] | None] = {}
for _card in PLAYER_DECK:
    for _pi in range(len(CARD_PLACEMENTS[_card])):
        ACTION_INDEX_MAP[ACTION_OFFSETS[_card] + _pi] = (_card, _pi)
ACTION_INDEX_MAP[WAIT_ACTION] = None
