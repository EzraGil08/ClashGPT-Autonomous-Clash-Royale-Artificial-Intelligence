"""
env.py — Gymnasium + MaskablePPO environment for the Clash Royale agent.
"""

from __future__ import annotations
import time
import numpy as np
import cv2 as cv
import mss
from pathlib import Path
from ultralytics import YOLO

import gymnasium as gym
from gymnasium import spaces

from misc_functions  import get_top_corner, get_elixir
from hand_detection  import get_hand
from tower_health    import get_tower_health, make_tower_state
from tracker         import CRTracker, Detection
from observation     import build_observation, OBS_SIZE
from reward          import compute_reward, reward_summary
from actions         import get_action_mask, execute_action, action_to_str
from constants       import N_ACTIONS, WAIT_ACTION, CARD_PLACEMENTS, ACTION_INDEX_MAP, ACTION_OFFSETS, CARD_COSTS, HAND_SLOT_COORDS

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE         = Path(r"C:\Users\Ezra\OneDrive\Desktop\Senior Research Project")
LBGG_WEIGHTS = BASE / r"YOLO\LBGG\LBGG_model\train\weights\best.pt"
SARG_WEIGHTS = BASE / r"YOLO\SARG\SARG_model\train\weights\best.pt"

# ── Constants ─────────────────────────────────────────────────────────────────

GAME_W           = 561
GAME_H           = 998
CONF_THRESH      = 0.35
IOU_NMS          = 0.45
HP_DELTA_THRESHOLD = 0.5
YOLO_SKIP_FRAMES = 3
ACTION_COOLDOWN  = 0.15

# Steps to wait after a damage spell before evaluating miss/hit.
# At ~0.3-0.5s per step this is ~2-3s — enough for arrows/snowball to land.
SPELL_EVAL_STEPS = 6

_DAMAGE_SPELLS = {"arrows", "snowball"}

# ── Explicit team sets ────────────────────────────────────────────────────────

_ALLY_CLASSES  = {
    "giant", "bowler", "witch", "guard", "minion", "skeleton",
    "graveyard","arrows", "snowball"
}
_ENEMY_CLASSES = {
    "princess", "ice_spirit", "rocket", "goblin_barrel",
    "knight", "log", "cannon",
    "goblin", "spear_goblin", "goblin_gang",
}


# ── Team assignment ───────────────────────────────────────────────────────────

def _assign_team(cls: str, norm_y: float) -> str:
    if cls in _ALLY_CLASSES:
        return "ally"
    if cls in _ENEMY_CLASSES:
        return "enemy"
    print(f"[WARN] _assign_team: unknown class '{cls}', defaulting to ally")
    return "ally"


# ── HP noise filter ───────────────────────────────────────────────────────────

def _filter_hp_noise(prev, curr, threshold=HP_DELTA_THRESHOLD):
    filtered = dict(curr)
    for k in prev:
        if k not in curr:
            continue
        if curr[k] <= 0.0:
            filtered[k] = 0.0
        elif abs(curr[k] - prev[k]) < threshold:
            filtered[k] = prev[k]
    return filtered


# ── Termination helpers ───────────────────────────────────────────────────────

def _ally_all_dead(health):
    return (
        health.get("ally_left",  100) <= 0 and
        health.get("ally_right", 100) <= 0 and
        health.get("ally_king",  100) <= 0
    )

def _enemy_all_dead(health):
    return (
        health.get("enemy_left",  100) <= 0 and
        health.get("enemy_right", 100) <= 0 and
        health.get("enemy_king",  100) <= 0
    )


# ── Environment ───────────────────────────────────────────────────────────────

class CREnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 0}

    def __init__(self, calibration_region, render_mode=None):
        super().__init__()

        self.left, self.top, self.cap_w, self.cap_h = calibration_region
        self.render_mode    = render_mode
        self._capture_offset = (self.left, self.top)

        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        print("Loading YOLO models …")
        self._lbgg = YOLO(str(LBGG_WEIGHTS))
        self._sarg = YOLO(str(SARG_WEIGHTS))
        print("Models loaded.")

        self._tracker = CRTracker(max_age=4, min_hits=2, iou_threshold=0.25)
        self._sct     = mss.mss()
        self._monitor = {"left": self.left, "top": self.top,
                         "width": self.cap_w, "height": self.cap_h}

        # Episode state — initialised properly in reset()
        self._corner:             tuple   = (0, 0)
        self._last_health:        dict    = {}
        self._fallback_counters:  dict    = {}
        self._tower_state:        dict    = {}
        self._last_obs:           np.ndarray = np.zeros(OBS_SIZE, dtype=np.float32)
        self._last_mask:          np.ndarray = np.ones(N_ACTIONS, dtype=bool)
        self._last_hand:          list    = []
        self._last_tracks:        list    = []
        self._last_track_ids:     set     = set()
        self._done:               bool    = False
        self._force_done:         bool    = False
        self._won:                bool    = False
        self._step_count:         int     = 0
        self._ep_reward:          float   = 0.0
        self._yolo_frame:         int     = 0
        self._last_elixir:        float   = 0.0

        # Delayed spell evaluation state
        self._pending_spell:      dict | None = None  # {card, lane, countdown}
        self._spell_disappeared:  set         = set() # enemy IDs gone since spell played
        self._spell_prev_map:     dict        = {}    # id→TrackResult at spell-play time
        # Barrel tracking: steps since barrel landed on each side (None = no recent barrel)
        self._recent_barrels:     dict        = {"left": None, "right": None}
        self._barrel_response_steps: int | None = None
        self._barrel_response_side:  str | None = None
        # Barrel intercept override state machine
        # States: None=idle, 'tracking'=barrel in flight, 'fire'=barrel landed
        self._barrel_intercept_state: str | None = None
        self._barrel_intercept_lane:  str | None = None  # locked when barrel lands
        self._barrel_intercept_card:  str | None = None  # card chosen to play
        self._barrel_intercept_timer: int        = 0     # steps in current state

    # ── Barrel intercept override ────────────────────────────────────────────

    def _barrel_intercept_override(self, action: int) -> int:
        """
        State machine that intercepts goblin barrel and forces a defensive spell
        or cheapest card play when the barrel lands.

        States
        ------
        None      : idle — no barrel detected, pass action through unchanged
        'tracking': barrel in flight — saving elixir, lane not yet locked
        'fire'    : barrel just landed — lane locked, play card ASAP
                    aborts after FIRE_TIMEOUT steps (~3s) if still can't afford

        Lane is locked when the barrel DISAPPEARS (lands), not when first seen.
        This avoids playing on the wrong side when barrel spawns near centre.

        Returns the (possibly overridden) action index.
        """
        FIRE_TIMEOUT = 20   # ~3s at ~0.3s/step — abort if can't fire in time

        hand   = self._last_hand
        elixir = self._last_elixir
        tracks = self._last_tracks

        # ── Detect new barrel in flight ───────────────────────────────────────
        if self._barrel_intercept_state is None:
            for t in tracks:
                if t.team == "enemy" and t.cls == "goblin_barrel":
                    # Don't lock lane yet — just note a barrel exists and pick card.
                    # Lane is locked when barrel disappears (landing position is reliable).
                    card = self._choose_intercept_card(hand, None)
                    if card is not None:
                        self._barrel_intercept_state = "tracking"
                        self._barrel_intercept_lane  = None   # locked on disappearance
                        self._barrel_intercept_card  = card
                        self._barrel_intercept_timer = 0
                        print(f"[barrel] TRACKING — barrel detected, will play {card} on landing")
                    break

        # ── Check if tracked barrel has landed (disappeared) ──────────────────
        # Lock lane based on where barrel was last seen before disappearing.
        if self._barrel_intercept_state == "tracking":
            self._barrel_intercept_timer += 1
            barrel_track = next(
                (t for t in tracks if t.team == "enemy" and t.cls == "goblin_barrel"),
                None
            )
            if barrel_track is None:
                # Barrel gone — lock lane from _recent_barrels which was set
                # by the just_gone detection in step() using last known position
                # Pick whichever side just had a barrel land (set in step())
                lane = None
                for side in ("left", "right"):
                    if self._recent_barrels.get(side) == 0:  # just landed this step
                        lane = side
                        break
                if lane is None:
                    # Fallback: use last seen position of any recent barrel
                    for side in ("left", "right"):
                        if self._recent_barrels.get(side) is not None:
                            lane = side
                            break
                if lane is None:
                    lane = "right"  # shouldn't happen, safe default
                self._barrel_intercept_lane  = lane
                self._barrel_intercept_state = "fire"
                self._barrel_intercept_timer = 0
                print(f"[barrel] FIRE — barrel landed in {lane} lane")

        # ── TRACKING: force wait to save elixir ───────────────────────────────
        if self._barrel_intercept_state == "tracking":
            return WAIT_ACTION

        # ── FIRE: barrel landed, play card ASAP ───────────────────────────────
        if self._barrel_intercept_state == "fire":
            self._barrel_intercept_timer += 1
            card = self._barrel_intercept_card
            lane = self._barrel_intercept_lane
            cost = CARD_COSTS.get(card, 3)

            # Timeout — abort if took too long (~3s)
            if self._barrel_intercept_timer > FIRE_TIMEOUT:
                print(f"[barrel] TIMEOUT — could not fire {card} in time, aborting")
                self._barrel_intercept_state = None
                self._barrel_intercept_lane  = None
                self._barrel_intercept_card  = None
                return action

            # Card left hand — abort
            if card not in (self._last_hand or []):
                print(f"[barrel] ABORT — {card} no longer in hand")
                self._barrel_intercept_state = None
                self._barrel_intercept_lane  = None
                self._barrel_intercept_card  = None
                return action

            # Wait until affordable
            if elixir < cost:
                return WAIT_ACTION

            forced_action = self._tower_placement_action(card, lane)

            if forced_action is not None:
                # Spell — use existing action index so PPO learns from it
                print(f"[barrel] EXECUTING spell — {card} at {lane} tower position")
                self._barrel_intercept_state = None
                self._barrel_intercept_lane  = None
                self._barrel_intercept_card  = None
                return forced_action
            else:
                # Non-spell — direct pyautogui click, return WAIT to PPO
                cx, cy   = self._corner
                ox, oy   = self._capture_offset
                tx, ty   = (134, 650) if lane == "left" else (428, 650)
                slot_idx = next(
                    (i for i, c in enumerate(self._last_hand[:4]) if c == card),
                    None
                )
                if slot_idx is not None:
                    sx, sy = HAND_SLOT_COORDS[slot_idx]
                    import pyautogui
                    pyautogui.click(ox + cx + sx, oy + cy + sy)
                    time.sleep(0.05)
                    pyautogui.click(ox + cx + tx, oy + cy + ty)
                    print(f"[barrel] EXECUTING direct — {card} at {lane} ({tx},{ty})")
                self._barrel_intercept_state = None
                self._barrel_intercept_lane  = None
                self._barrel_intercept_card  = None
                return WAIT_ACTION

        return action

    def _choose_intercept_card(self, hand: list, lane) -> str | None:
        """
        Pick the best card to intercept a goblin barrel.
        Priority: snowball > arrows > cheapest other card in hand.
        lane param unused but kept for API consistency.
        Returns card name or None if hand is empty.
        """
        hand_set = set(hand) - {"empty"}
        if not hand_set:
            return None
        if "snowball" in hand_set:
            return "snowball"
        if "arrows" in hand_set:
            return "arrows"
        
        return min(hand_set, key=lambda c: CARD_COSTS.get(c, 99))

    def _tower_placement_action(self, card: str, lane: str) -> int | None:
        """
        For spells (arrows/snowball): return the existing action index at
        placement_idx 2 (left, 132,611) or 3 (right, 427,610) — these are
        in the PPO action space so the agent learns from them normally.

        For all other cards: return None — caller will do a direct pyautogui
        click to (134,650) left or (428,650) right, bypassing the action space.
        PPO sees WAIT_ACTION for these plays.
        """
        if card in _DAMAGE_SPELLS:
            pi = 2 if lane == "left" else 3
            placements = CARD_PLACEMENTS.get(card, [])
            if pi < len(placements):
                return ACTION_OFFSETS[card] + pi
        return None   # non-spell: caller handles direct click

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _grab_frame(self):
        return cv.cvtColor(np.array(self._sct.grab(self._monitor)), cv.COLOR_BGRA2BGR)

    def _process_frame(self, frame, run_yolo=True):
        cx, cy = self._corner
        game_region = frame[max(cy,0):cy+GAME_H, max(cx,0):cx+GAME_W]
        gh, gw = game_region.shape[:2]

        if run_yolo:
            res_lbgg = self._lbgg.predict(game_region, conf=CONF_THRESH, iou=IOU_NMS,
                                           verbose=False, device=0)[0]
            res_sarg = self._sarg.predict(game_region, conf=CONF_THRESH, iou=IOU_NMS,
                                           verbose=False, device=0)[0]
            detections = []
            for res in (res_lbgg, res_sarg):
                names = res.names
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf   = float(box.conf[0])
                    cls    = names[int(box.cls[0])]
                    norm_y = ((y1+y2)/2.0)/gh if gh > 0 else 0.5
                    team   = _assign_team(cls, norm_y)
                    detections.append(Detection(x1=x1,y1=y1,x2=x2,y2=y2,
                                                conf=conf,cls=cls,team=team))
            tracks = self._tracker.update(detections)
        else:
            tracks = self._tracker.update([])

        hand, next_card = get_hand(frame, self._corner)
        elixir          = get_elixir(frame, self._corner, self._last_elixir)
        health, _       = get_tower_health(frame, self._corner,
                                           self._last_health, self._tower_state)
        obs  = build_observation(tracks, hand, next_card, elixir, health)
        mask = get_action_mask(hand, elixir)
        return obs, mask, hand, health, tracks, next_card, elixir, game_region

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._tracker.reset()
        self._done              = False
        self._force_done        = False
        self._won               = False
        self._step_count        = 0
        self._ep_reward         = 0.0
        self._yolo_frame        = 0
        self._last_elixir       = 0.0
        self._fallback_counters = {}
        self._tower_state       = make_tower_state()
        self._last_health       = {
            "ally_left": 100.0, "ally_right": 100.0,
            "enemy_left": 100.0, "enemy_right": 100.0,
            "ally_king": 100.0, "enemy_king": 100.0,
        }
        self._last_tracks       = []
        self._last_track_ids    = set()
        self._pending_spell     = None
        self._spell_disappeared = set()
        self._spell_prev_map    = {}
        self._recent_barrels    = {"left": None, "right": None}
        self._barrel_response_steps = None
        self._barrel_response_side  = None
        self._barrel_intercept_state = None
        self._barrel_intercept_lane  = None
        self._barrel_intercept_card  = None
        self._barrel_intercept_timer = 0

        # Retry get_top_corner until the game frame is visible.
        # If the screen is still on the end-game banner, corner will be None.
        for _attempt in range(20):
            frame        = self._grab_frame()
            self._corner = get_top_corner(frame)
            if self._corner is not None:
                break
            print(f"[reset] waiting for game frame… (attempt {_attempt+1}/20)")
            time.sleep(1.0)
        if self._corner is None:
            raise RuntimeError("reset(): get_top_corner() returned None after 20s — "
                               "is the game visible on screen?")

        obs, mask, hand, health, tracks, next_card, elixir, _ = \
            self._process_frame(frame, run_yolo=True)

        self._last_health    = health
        self._last_elixir    = elixir
        self._last_obs       = obs
        self._last_mask      = mask
        self._last_hand      = hand
        self._last_tracks    = tracks
        self._last_track_ids = {getattr(t,"track_id",None) for t in tracks} - {None}

        return obs, {"step": 0, "hand": hand, "next_card": next_card,
                     "elixir": elixir, "health": health,
                     "action_str": "", "reward_str": ""}

    def step(self, action: int):
        if self._done and not self._force_done:
            raise AssertionError("step() called on finished episode — call reset() first")

        prev_health    = dict(self._last_health)
        prev_tracks    = self._last_tracks
        prev_track_ids = self._last_track_ids
        prev_elixir    = self._last_elixir
        played_card    = None
        played_lane    = None
        played_y       = None

        # ── Barrel intercept override ─────────────────────────────────────────
        # Detects flying goblin_barrel, saves elixir, fires spell/cheapest card
        # when barrel lands. Overrides PPO action during the intercept sequence.
        action = self._barrel_intercept_override(action)

        # ── Execute action ────────────────────────────────────────────────────
        if action != WAIT_ACTION:
            info_pair = ACTION_INDEX_MAP.get(action)
            if info_pair is not None:
                played_card, placement_idx = info_pair
                px = CARD_PLACEMENTS[played_card][placement_idx][0]
                played_lane = "left" if px < 280 else "right"
            execute_action(action, self._last_hand, self._corner, self._capture_offset)
            # Compute placement y for reward purpose signals
            if info_pair is not None:
                played_y = float(CARD_PLACEMENTS[played_card][placement_idx][1])
            else:
                played_y = None

            # Start delayed spell window immediately on spell play — before
            # any frame capture so timing is wall-clock accurate.
            if played_card in _DAMAGE_SPELLS and self._pending_spell is None:
                self._pending_spell     = {
                    "card":      played_card,
                    "lane":      played_lane,
                    "countdown": SPELL_EVAL_STEPS,
                }
                self._spell_disappeared = set()
                # Snapshot enemy track map at the moment the spell is played
                self._spell_prev_map = {
                    getattr(t, "track_id", None): t
                    for t in self._last_tracks
                    if t.team == "enemy"
                }
                self._spell_prev_map.pop(None, None)

            # Barrel response — check if spell placed at tower-defense position
            # after a barrel recently landed on that side.
            # Tower-defense placement indices: 2=(132,611) left, 3=(427,610) right
            if (played_card in _DAMAGE_SPELLS
                    and info_pair is not None
                    and placement_idx in (2, 3)):
                tower_side = "left" if placement_idx == 2 else "right"
                barrel_steps = self._recent_barrels.get(tower_side)
                if barrel_steps is not None:
                    self._barrel_response_steps = barrel_steps
                    self._barrel_response_side  = tower_side
                    self._recent_barrels[tower_side] = None  # consume it
                else:
                    self._barrel_response_steps = None
                    self._barrel_response_side  = None
            else:
                self._barrel_response_steps = None
                self._barrel_response_side  = None

            time.sleep(ACTION_COOLDOWN)

        # ── Capture next frame ────────────────────────────────────────────────
        self._yolo_frame += 1
        run_yolo = (self._yolo_frame % YOLO_SKIP_FRAMES == 0)

        frame = self._grab_frame()
        obs, mask, hand, health, tracks, next_card, elixir, game_region = \
            self._process_frame(frame, run_yolo=run_yolo)

        health = _filter_hp_noise(prev_health, health)

        # ── Accumulate disappeared enemy IDs for pending spell ────────────────
        # Runs every step regardless of YOLO skip — tracker coasts on old
        # predictions so track IDs still disappear on non-YOLO frames.
        curr_track_ids = {getattr(t,"track_id",None) for t in tracks} - {None}
        just_gone      = prev_track_ids - curr_track_ids

        # Track barrel landings — note which side a barrel just disappeared on
        _prev_map_quick = {getattr(t, "track_id", None): t for t in prev_tracks}
        for tid in just_gone:
            pt = _prev_map_quick.get(tid)
            if pt is not None and pt.team == "enemy" and pt.cls == "goblin_barrel":
                centre_x = (pt.x1 + pt.x2) / 2.0
                side = "left" if centre_x < 280 else "right"
                self._recent_barrels[side] = 0   # landed this step

        # Age barrel entries, expire after 10 steps (~3s)
        for side in ("left", "right"):
            if self._recent_barrels[side] is not None:
                self._recent_barrels[side] += 1
                if self._recent_barrels[side] > 10:
                    self._recent_barrels[side] = None

        if self._pending_spell is not None:
            for tid in just_gone:
                pt = _prev_map_quick.get(tid)
                if pt is not None and pt.team == "enemy":
                    self._spell_disappeared.add(tid)
            self._pending_spell["countdown"] -= 1

        # ── Evaluate spell when countdown expires ─────────────────────────────
        spell_card             = None
        spell_lane             = None
        spell_disappeared_ids  = set()
        spell_prev_map         = {}

        if self._pending_spell is not None and self._pending_spell["countdown"] <= 0:
            spell_card            = self._pending_spell["card"]
            spell_lane            = self._pending_spell["lane"]
            spell_disappeared_ids = set(self._spell_disappeared)
            spell_prev_map        = dict(self._spell_prev_map)

            if len(spell_disappeared_ids) == 0:
                print(f"[spell] MISS — {spell_card} ({spell_lane}), "
                      f"0 enemy tracks gone in {SPELL_EVAL_STEPS} steps")
            else:
                print(f"[spell] HIT  — {spell_card} ({spell_lane}), "
                      f"{len(spell_disappeared_ids)} enemy track(s) gone")

            self._pending_spell     = None
            self._spell_disappeared = set()
            self._spell_prev_map    = {}

        # ── Done detection ────────────────────────────────────────────────────
        terminated = self._force_done

        # ── Reward ────────────────────────────────────────────────────────────
        reward = compute_reward(
            prev_health           = prev_health,
            curr_health           = health,
            done                  = terminated,
            won                   = self._won,
            played_card           = played_card,
            played_lane           = played_lane,
            played_y              = played_y,
            spell_card            = spell_card,
            spell_lane            = spell_lane,
            spell_disappeared_ids = spell_disappeared_ids,
            spell_prev_map        = spell_prev_map,
            barrel_response_steps = self._barrel_response_steps,
            elixir                = prev_elixir,
            tracks                = tracks,
            prev_track_ids        = prev_track_ids,
            prev_tracks           = prev_tracks,
            hand                  = hand,
        )
        rew_str = reward_summary(
            prev_health           = prev_health,
            curr_health           = health,
            done                  = terminated,
            won                   = self._won,
            played_card           = played_card,
            played_lane           = played_lane,
            played_y              = played_y,
            spell_card            = spell_card,
            spell_lane            = spell_lane,
            spell_disappeared_ids = spell_disappeared_ids,
            spell_prev_map        = spell_prev_map,
            barrel_response_steps = self._barrel_response_steps,
            elixir                = prev_elixir,
            tracks                = tracks,
            prev_track_ids        = prev_track_ids,
            prev_tracks           = prev_tracks,
            hand                  = hand,
        )

        # Clear barrel response — only fires on the step the spell is played
        self._barrel_response_steps = None
        self._barrel_response_side  = None

        # ── Update state ──────────────────────────────────────────────────────
        self._last_health    = health
        self._last_elixir    = elixir
        self._last_obs       = obs
        self._last_mask      = mask
        self._last_hand      = hand
        self._last_tracks    = tracks
        self._last_track_ids = curr_track_ids
        self._step_count    += 1
        self._ep_reward     += reward

        if terminated:
            self._done       = True
            self._force_done = False

        if self.render_mode == "human":
            self._render_frame(game_region, tracks, health, hand,
                               next_card, elixir, action, reward, run_yolo)

        return obs, reward, terminated, False, {
            "step":         self._step_count,
            "hand":         hand,
            "next_card":    next_card,
            "elixir":       elixir,
            "health":       health,
            "action_str":   action_to_str(action),
            "reward_str":   rew_str,
            "ep_reward":    self._ep_reward,
            "yolo_ran":     run_yolo,
        }

    def action_masks(self):
        return self._last_mask

    def force_done(self, won: bool):
        self._force_done = True
        self._won        = won

    # ── Render ────────────────────────────────────────────────────────────────

    def _render_frame(self, game_region, tracks, health, hand,
                      next_card, elixir, action, reward, run_yolo=True):
        vis  = game_region.copy()
        FONT = cv.FONT_HERSHEY_SIMPLEX
        for t in tracks:
            color = (80,220,80) if t.team=="ally" else (60,60,220)
            x1,y1,x2,y2 = int(t.x1),int(t.y1),int(t.x2),int(t.y2)
            cv.rectangle(vis,(x1,y1),(x2,y2),color,2)
            cv.putText(vis,f"{t.cls}#{t.track_id}",(x1,max(y1-4,12)),
                       FONT,0.4,color,1,cv.LINE_AA)
        yolo_tag = "[YOLO]" if run_yolo else "[skip]"
        spell_tag = f" [spell:{self._pending_spell['card']}:{self._pending_spell['countdown']}]" \
                    if self._pending_spell else ""
        lines = [
            f"Step {self._step_count}  Elixir:{elixir:.1f}  {yolo_tag}{spell_tag}",
            f"Hand: {' '.join(hand)}",
            f"Next: {next_card}",
            f"Action: {action_to_str(action)}",
            f"Rew: {reward:+.3f}  Total:{self._ep_reward:+.2f}",
            f"HP AL:{health.get('ally_left',0):.0f} "
            f"AR:{health.get('ally_right',0):.0f} "
            f"EL:{health.get('enemy_left',0):.0f} "
            f"ER:{health.get('enemy_right',0):.0f} "
            f"AK:{health.get('ally_king',0):.0f} "
            f"EK:{health.get('enemy_king',0):.0f}",
        ]
        y = 16
        for line in lines:
            cv.putText(vis,line,(5,y),FONT,0.42,(0,0,0),3,cv.LINE_AA)
            cv.putText(vis,line,(5,y),FONT,0.42,(255,255,255),1,cv.LINE_AA)
            y += 16
        cv.imshow("CR Agent", vis)
        cv.waitKey(1)

    def close(self):
        self._sct.close()
        cv.destroyAllWindows()