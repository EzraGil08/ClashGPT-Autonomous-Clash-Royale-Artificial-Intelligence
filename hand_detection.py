import cv2 as cv
import numpy as np
from pathlib import Path

BASE = r"C:\Users\Ezra\OneDrive\Desktop\Senior Research Project\Images"
CARD_NAMES = ["arrows", "bowler", "giant", "graveyard", "guards", "minions", "snowball", "witch","empty"]

card_files      = [f"{BASE}\\cards_in_hand\\{name}.png"  for name in CARD_NAMES]
mini_card_files = [f"{BASE}\\mini_cards\\mini_{name}.png" for name in CARD_NAMES[:-1]]  # exclude "empty"

def get_hand(img, corner):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _x, _y = corner
    region = img[_y+820 : _y+951, _x+124 : _x+541]
    method = cv.TM_CCOEFF_NORMED

    best_matches = []

    for i, f in enumerate(card_files):
        card = cv.imread(f, cv.IMREAD_GRAYSCALE)
        if card is None:
            print(f"Warning: could not load card image: {f}")
            continue
        if region.shape[0] < card.shape[0] or region.shape[1] < card.shape[1]:
            return ["empty", "empty", "empty", "empty"], None
        result = cv.matchTemplate(region, card, method)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        best_matches.append((max_val, max_loc, CARD_NAMES[i]))

    best_matches.sort(key=lambda x: x[0], reverse=True)

    def select_spread_matches(matches, n=4, min_x_dist=50):
        selected = []
        for match in matches:
            _, (x, y), _ = match
            if all(abs(x - sx) >= min_x_dist for _, (sx, _), _ in selected):
                selected.append(match)
            if len(selected) == n:
                break
        return selected

    top_four = select_spread_matches(best_matches)
    top_four.sort(key=lambda x: x[1][0])
    hand = [name for _, _, name in top_four]

    if len(hand) < 4:
        locations = [16,122,227,332]
        for i in range(len(hand)):
            if abs(top_four[i][1][0] - locations[i]) > 5:
                hand.insert(i, "empty")
                break

    # next card detection
    next_card_region = img[_y+930 : _y+986, _x+29 : _x+75]
    best_next_match = (0, None, None)

    for i, f in enumerate(mini_card_files):
        card = cv.imread(f, cv.IMREAD_GRAYSCALE)
        if card is None:
            print(f"Warning: could not load card image: {f}")
            continue
        if next_card_region.shape[0] < card.shape[0] or next_card_region.shape[1] < card.shape[1]:
            return ["empty", "empty", "empty", "empty"], None
        result = cv.matchTemplate(next_card_region, card, method)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        if max_val > best_next_match[0]:
            best_next_match = (max_val, max_loc, CARD_NAMES[i])

    next_card = best_next_match[2]
    return hand, next_card


# # ── TEST ─────────────────────────────────────────────────────
# from misc_functions import get_top_corner

# GAME_FILE = r"C:\Users\Ezra\OneDrive\Desktop\Senior Research Project\Images\test_images\Game6.png"
# GAME = cv.imread(GAME_FILE)

# corner = get_top_corner(GAME)
# hand, next_card = get_hand(GAME, corner)
# print("Hand:", hand)
# print("Next card:", next_card)