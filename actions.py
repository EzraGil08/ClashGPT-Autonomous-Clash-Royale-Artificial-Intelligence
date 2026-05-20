"""
actions.py — action masking and execution for the CR agent.

Public API
----------
get_action_mask(hand, elixir)  ->  np.ndarray shape (N_ACTIONS,) dtype bool
    True  = action is legal this step.
    False = action is masked out (MaskablePPO will never sample it).

    Masking rules:
      • A card's entire placement block is masked if:
          (a) the card is not currently in hand[0:4], OR
          (b) the player has less elixir than the card's cost.
      • WAIT_ACTION is always True.

execute_action(action_idx, hand, corner)  ->  bool
    Translates a flat action index into PyAutoGUI clicks and executes it.
    Returns True if an actual card was played, False for wait.
    `hand`   : list[str] of length 4, left-to-right (from get_hand()).
    `corner` : (cx, cy) pixel offset returned by get_top_corner().
"""

import time
import numpy as np
import pyautogui

from constants import (
    CARD_COSTS,
    CARD_PLACEMENTS,
    HAND_SLOT_COORDS,
    ACTION_OFFSETS,
    ACTION_INDEX_MAP,
    WAIT_ACTION,
    N_ACTIONS,
    PLAYER_DECK,
)

# PyAutoGUI safety — lower pause reduces latency during play
pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True


# ── Action mask ───────────────────────────────────────────────────────────────

def get_action_mask(hand: list[str], elixir: float) -> np.ndarray:
    """
    Parameters
    ----------
    hand   : list of exactly 4 card name strings (may include "empty").
    elixir : current elixir as a float in [0, 10].

    Returns
    -------
    Boolean mask of shape (N_ACTIONS,).
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)

    hand_set = set(hand) - {"empty"}

    for card in PLAYER_DECK:
        block_start = ACTION_OFFSETS[card]
        block_end   = block_start + len(CARD_PLACEMENTS[card])

        in_hand      = card in hand_set
        can_afford   = elixir >= CARD_COSTS[card]

        if in_hand and can_afford:
            mask[block_start:block_end] = True
        # else: entire placement block stays False

    # Wait is always legal
    mask[WAIT_ACTION] = True

    return mask


# ── Action execution ──────────────────────────────────────────────────────────

def execute_action(
    action_idx: int,
    hand: list[str],
    corner: tuple[int, int],
    capture_offset: tuple[int, int] = (0, 0),  # (left, top) from calibrate()
) -> bool:
    if action_idx == WAIT_ACTION:
        return False

    card_placement = ACTION_INDEX_MAP.get(action_idx)
    if card_placement is None:
        return False

    card, placement_idx = card_placement
    cx, cy = corner
    ox, oy = capture_offset  # screen offset of the capture region

    slot_idx = None
    for i, c in enumerate(hand[:4]):
        if c == card:
            slot_idx = i
            break

    if slot_idx is None:
        return False

    # Hand slot click — corner-relative + capture offset = screen absolute
    sx, sy = HAND_SLOT_COORDS[slot_idx]
    pyautogui.click(ox + cx + sx, oy + cy + sy)
    time.sleep(0.05)

    # Placement click
    px, py = CARD_PLACEMENTS[card][placement_idx]
    pyautogui.click(ox + cx + px, oy + cy + py)

    return True


# ── Utility ───────────────────────────────────────────────────────────────────

def action_to_str(action_idx: int) -> str:
    """Human-readable label for logging / dashboard."""
    if action_idx == WAIT_ACTION:
        return "WAIT"
    info = ACTION_INDEX_MAP.get(action_idx)
    if info is None:
        return f"UNKNOWN({action_idx})"
    card, pi = info
    x, y = CARD_PLACEMENTS[card][pi]
    return f"{card}@({x},{y})"
