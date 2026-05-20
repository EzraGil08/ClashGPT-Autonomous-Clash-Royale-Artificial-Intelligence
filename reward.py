"""
reward.py — hybrid dense/sparse reward for the CR agent.

Key design principles
---------------------
- Purpose over noise: rewards are for *good decisions*, not just outcomes
- Troop kills removed except princess (high value target)
- Ally troop death penalty removed entirely
- New purpose rewards:
    defensive_placement  : defensive unit played near y=520 when enemy approaching
    wrong_placement_deep : penalty for defensive unit placed too deep (y=756+)
    nothing_on_screen    : penalty for playing non-giant when board is empty
    wrong_lane           : penalty for playing into empty lane when ally lane active
    witch_behind_giant   : large bonus
    graveyard_in_push    : massively scaled up, giant-over-bridge is huge
    behind_giant         : kept, moderate

Spell evaluation
----------------
Delayed: env.py accumulates disappeared enemy IDs for SPELL_EVAL_STEPS after
a spell is played, then passes spell_card/spell_lane/spell_disappeared_ids/
spell_prev_map on the eval step.

Public API
----------
compute_reward(
    prev_health, curr_health, done, won,
    played_card, played_lane, played_y,     # immediate action
    spell_card, spell_lane,                 # delayed spell eval
    spell_disappeared_ids, spell_prev_map,
    barrel_response_steps,
    elixir, tracks, prev_track_ids, prev_tracks, hand,
) -> float
"""

from __future__ import annotations
from dataclasses import dataclass, field
from constants import CARD_COSTS, CARD_PLACEMENTS

# ── Constants ─────────────────────────────────────────────────────────────────

GAME_W               = 420.0
GAME_H               = 998.0
BRIDGE_NORM_Y        = 430.0 / 998.0   # ≈ 0.431
TOWER_DANGER_NORM_Y  = 0.75
BARREL_DANGER_NORM_Y = 0.60
ENEMY_APPROACH_Y     = 340.0           # enemy troops past this y are "approaching"

ELIXIR_PATIENCE      = 1.5

_DAMAGE_SPELLS     = {"arrows", "snowball"}
_GOBLIN_CLASSES    = {"goblin", "spear_goblin", "goblin_gang", "goblin_barrel"}
_DEFENSIVE_UNITS   = {"bowler", "witch", "guards", "minions"}
# Good placement y range for defensive units (~520 area, placement_idx 0,1,4,5)
_DEF_GOOD_Y_MIN    = 450
_DEF_GOOD_Y_MAX    = 560
# Bad placement y (too deep, placement_idx 2,3 at y=756/760)
_DEF_BAD_Y_MIN     = 700


# ── Reward weights ────────────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    # ── Tower HP (per 1% HP) ──────────────────────────────────────────────────
    hp_dealt_per_pct:               float = 0.02
    hp_taken_per_pct:               float = -0.04

    # ── Dense: defense pressure (per enemy troop near our towers) ─────────────
    defense_penalty_per_troop:      float = -0.01

    # ── Troop kills — only princess now ──────────────────────────────────────
    princess_kill_bonus:            float = 1.5

    # ── Idle / timing ─────────────────────────────────────────────────────────
    idle_penalty:                   float = 0.03
    wait_forced_save_bonus:         float = 0.01
    play_too_fast_penalty:          float = -0.20

    # ── Purpose: defensive placement ─────────────────────────────────────────
    # Bonus per enemy troop past y=340 in same lane when defensive unit placed
    defensive_purpose_per_troop:    float = 0.5
    # Bonus for placing in good y range (450-560)
    defensive_good_position:        float = 0.5
    # Penalty for placing too deep (y > 700)
    defensive_bad_position:         float = -0.5

    # ── Purpose: wrong lane penalty ───────────────────────────────────────────
    # Fires when card played in lane with no ally troops but other lane has some
    wrong_lane_penalty:             float = -0.5

    # ── Purpose: nothing on screen penalty ────────────────────────────────────
    # Fires when non-giant played with zero tracks on screen
    nothing_on_screen_penalty:      float = -0.5

    # ── Push bonuses ──────────────────────────────────────────────────────────
    graveyard_over_bridge_bonus:    float = 10.0   # giant over bridge + graveyard
    graveyard_in_push_bonus:        float = 5.0    # graveyard with giant not yet over bridge
    graveyard_push_bonus:           float = 2.0    # graveyard with any ally in enemy half
    witch_behind_giant_bonus:       float = 2.0    # witch specifically behind giant
    behind_giant_bonus:             float = 0.5    # any other card behind giant
    behind_giant_per_troop:         float = 0.1

    # ── Spell targeting ───────────────────────────────────────────────────────
    spell_miss_penalty:             float = -5.0
    spell_killed_goblin_bonus:      float = 0.4
    spell_killed_princess_bonus:    float = 2.0
    goblin_barrel_intercept_bonus:  float = 5.0
    barrel_response_bonus_max:      float = 6.0
    barrel_response_window:         int   = 10

    # ── Tower events ──────────────────────────────────────────────────────────
    enemy_tower_bonus:              float = 3.0
    ally_tower_penalty:             float = -3.0

    # ── Game over ─────────────────────────────────────────────────────────────
    win_bonus:                      float = 0.0
    loss_penalty:                   float = 0.0


WEIGHTS = RewardWeights()

_ENEMY_TOWERS = ("enemy_left", "enemy_right")
_ALLY_TOWERS  = ("ally_left",  "ally_right")
_ALL_TOWERS   = _ENEMY_TOWERS + _ALLY_TOWERS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hp_delta(prev, curr):
    return {k: curr.get(k, 0.0) - prev.get(k, 0.0) for k in _ALL_TOWERS}

def _just_destroyed(prev, curr, key):
    return prev.get(key, 100.0) > 0.0 and curr.get(key, 100.0) <= 0.0

def _troop_center(t):
    return ((t.x1 + t.x2) / 2.0) / GAME_W, ((t.y1 + t.y2) / 2.0) / GAME_H

def _troop_center_px(t):
    """Returns pixel coords (not normalized)."""
    return (t.x1 + t.x2) / 2.0, (t.y1 + t.y2) / 2.0

def _in_lane(t, lane):
    cx, _ = _troop_center_px(t)
    return (cx < 280) == (lane == "left")

def _troops_in_lane(tracks, team, lane, max_norm_y=None, min_norm_y=None):
    result = []
    for t in tracks:
        if t.team != team:
            continue
        norm_x, norm_y = _troop_center(t)
        if (norm_x < 0.5) != (lane == "left"):
            continue
        if max_norm_y is not None and norm_y >= max_norm_y:
            continue
        if min_norm_y is not None and norm_y < min_norm_y:
            continue
        result.append(t)
    return result

def _can_afford_any(hand, elixir):
    return any(CARD_COSTS.get(c, 99) <= elixir for c in hand if c != "empty")

def _enemy_approaching_in_lane(tracks, lane):
    """Enemy troops past ENEMY_APPROACH_Y (y > 340) in the given lane."""
    result = []
    for t in tracks:
        if t.team != "enemy":
            continue
        cx, cy = _troop_center_px(t)
        if (cx < 280) != (lane == "left"):
            continue
        if cy > ENEMY_APPROACH_Y:
            result.append(t)
    return result

def _ally_troops_in_lane(tracks, lane):
    """Any ally troop in the given lane."""
    return [t for t in tracks if t.team == "ally" and _in_lane(t, lane)]

def _other_lane(lane):
    return "right" if lane == "left" else "left"


# ── Public ────────────────────────────────────────────────────────────────────

def compute_reward(
    prev_health:            dict[str, float],
    curr_health:            dict[str, float],
    done:                   bool,
    won:                    bool,
    played_card:            str | None      = None,
    played_lane:            str | None      = None,
    played_y:               float | None    = None,   # pixel y of placement
    spell_card:             str | None      = None,
    spell_lane:             str | None      = None,
    spell_disappeared_ids:  set             = None,
    spell_prev_map:         dict            = None,
    barrel_response_steps:  int | None      = None,
    elixir:                 float           = 5.0,
    tracks:                 list            = None,
    prev_track_ids:         set             = None,
    prev_tracks:            list            = None,
    hand:                   list[str]       = None,
    weights:                RewardWeights   = WEIGHTS,
) -> float:

    if tracks is None:              tracks = []
    if prev_track_ids is None:      prev_track_ids = set()
    if prev_tracks is None:         prev_tracks = []
    if hand is None:                hand = []
    if spell_disappeared_ids is None: spell_disappeared_ids = set()
    if spell_prev_map is None:      spell_prev_map = {}

    reward = 0.0
    log    = []

    # ── Dense: tower HP ───────────────────────────────────────────────────────
    delta = _hp_delta(prev_health, curr_health)
    for key in _ENEMY_TOWERS:
        dmg = -delta[key]
        if dmg > 0:
            r = dmg * weights.hp_dealt_per_pct
            reward += r
            log.append(f"dealt({key[6:]}:{dmg:.1f}%,+{r:.3f})")
    for key in _ALLY_TOWERS:
        dmg = -delta[key]
        if dmg > 0:
            r = dmg * weights.hp_taken_per_pct
            reward += r
            log.append(f"took({key[4:]}:{dmg:.1f}%,{r:.3f})")

    # ── Dense: defense pressure ───────────────────────────────────────────────
    for t in tracks:
        if t.team != "enemy":
            continue
        _, norm_y = _troop_center(t)
        if norm_y > TOWER_DANGER_NORM_Y:
            reward += (norm_y - TOWER_DANGER_NORM_Y) * weights.defense_penalty_per_troop

    # ── Sparse: princess kill (only kept kill reward) ─────────────────────────
    current_ids = {getattr(t, "track_id", None) for t in tracks} - {None}
    disappeared = prev_track_ids - current_ids
    prev_map    = {getattr(t, "track_id", None): t for t in prev_tracks}
    prev_map.pop(None, None)
    for tid in disappeared:
        pt = prev_map.get(tid)
        if pt is not None and pt.team == "enemy" and pt.cls == "princess":
            reward += weights.princess_kill_bonus
            log.append(f"PRINCESS_KILLED(+{weights.princess_kill_bonus:.1f})")

    # ── Sparse: delayed spell evaluation ─────────────────────────────────────
    if spell_card is not None:
        if len(spell_disappeared_ids) == 0:
            reward += weights.spell_miss_penalty
            log.append(f"SPELL_MISS({spell_card},{weights.spell_miss_penalty:.2f})")
        else:
            spell_killed_goblins  = 0
            spell_killed_princess = False
            barrel_intercepted    = False
            for tid in spell_disappeared_ids:
                pt = spell_prev_map.get(tid)
                if pt is None:
                    continue
                if spell_lane is not None and not _in_lane(pt, spell_lane):
                    continue
                if pt.cls == "goblin_barrel":
                    _, norm_y = _troop_center(pt)
                    if norm_y > BARREL_DANGER_NORM_Y:
                        barrel_intercepted = True
                if pt.cls in _GOBLIN_CLASSES:
                    spell_killed_goblins += 1
                if pt.cls == "princess" and spell_card == "arrows":
                    spell_killed_princess = True
            if barrel_intercepted:
                reward += weights.goblin_barrel_intercept_bonus
                log.append(f"BARREL_INTERCEPT(+{weights.goblin_barrel_intercept_bonus:.1f})")
            if spell_killed_goblins > 0:
                r = spell_killed_goblins * weights.spell_killed_goblin_bonus
                reward += r
                log.append(f"spell_goblins(x{spell_killed_goblins},+{r:.3f})")
            if spell_killed_princess:
                reward += weights.spell_killed_princess_bonus
                log.append(f"SPELL_PRINCESS(+{weights.spell_killed_princess_bonus:.1f})")

    # ── Sparse: barrel response bonus ────────────────────────────────────────
    if barrel_response_steps is not None:
        frac = max(0.0, 1.0 - (barrel_response_steps / weights.barrel_response_window))
        r = weights.barrel_response_bonus_max * frac
        if r > 0:
            reward += r
            log.append(f"BARREL_RESPONSE(steps={barrel_response_steps},+{r:.2f})")

    # ── Sparse: idle / saving ─────────────────────────────────────────────────
    if played_card is None:
        if _can_afford_any(hand, elixir):
            reward += weights.idle_penalty
            log.append(f"idle({weights.idle_penalty:.3f})")
        elif elixir < 4.0:
            reward += weights.wait_forced_save_bonus
            # silent

    # ── Sparse: play-too-fast ─────────────────────────────────────────────────
    if played_card is not None:
        cost = CARD_COSTS.get(played_card, 3)
        if elixir < cost + ELIXIR_PATIENCE:
            reward += weights.play_too_fast_penalty
            log.append(f"too_fast({played_card}@{elixir:.1f},{weights.play_too_fast_penalty:.2f})")

    # ── Sparse: purpose rewards (only when a card was played) ─────────────────
    if played_card is not None and played_lane is not None:

        # ── Nothing on screen penalty ─────────────────────────────────────────
        # Non-giant played when board is completely empty — wasteful
        if played_card != "giant" and len(tracks) == 0:
            reward += weights.nothing_on_screen_penalty
            log.append(f"nothing_on_screen({played_card},{weights.nothing_on_screen_penalty:.2f})")

        # ── Wrong lane penalty ────────────────────────────────────────────────
        # Card played into lane with no allies, while other lane has allies
        if played_card != "giant":
            allies_this_lane  = _ally_troops_in_lane(tracks, played_lane)
            allies_other_lane = _ally_troops_in_lane(tracks, _other_lane(played_lane))
            if len(allies_this_lane) == 0 and len(allies_other_lane) > 0:
                reward += weights.wrong_lane_penalty
                log.append(f"wrong_lane({played_card},{weights.wrong_lane_penalty:.2f})")

        # ── Defensive placement purpose ───────────────────────────────────────
        if played_card in _DEFENSIVE_UNITS:
            approaching = _enemy_approaching_in_lane(tracks, played_lane)

            if len(approaching) > 0:
                # Reward scales with number of approaching troops
                r = len(approaching) * weights.defensive_purpose_per_troop
                reward += r
                log.append(f"def_purpose({played_card},{len(approaching)}troops,+{r:.3f})")

            # Position bonus/penalty
            if played_y is not None:
                if _DEF_GOOD_Y_MIN <= played_y <= _DEF_GOOD_Y_MAX:
                    reward += weights.defensive_good_position
                    log.append(f"def_good_pos(y={played_y:.0f},+{weights.defensive_good_position:.2f})")
                elif played_y >= _DEF_BAD_Y_MIN:
                    reward += weights.defensive_bad_position
                    log.append(f"def_bad_pos(y={played_y:.0f},{weights.defensive_bad_position:.2f})")

        # ── Graveyard push bonuses ────────────────────────────────────────────
        if played_card == "graveyard":
            giants_ob = [
                t for t in _troops_in_lane(tracks, "ally", played_lane, max_norm_y=BRIDGE_NORM_Y)
                if t.cls == "giant"
            ]
            if giants_ob:
                reward += weights.graveyard_over_bridge_bonus
                log.append(f"GY_BRIDGE(+{weights.graveyard_over_bridge_bonus:.1f},"
                            f"{len(giants_ob)}g_over_bridge)")
            else:
                # Giant in same lane but not over bridge yet — still great
                giants_in_lane = [
                    t for t in _troops_in_lane(tracks, "ally", played_lane)
                    if t.cls == "giant"
                ]
                if giants_in_lane:
                    reward += weights.graveyard_in_push_bonus
                    log.append(f"GY_IN_PUSH(+{weights.graveyard_in_push_bonus:.1f},"
                                f"{len(giants_in_lane)}g_in_lane)")
                else:
                    pushing = _troops_in_lane(tracks, "ally", played_lane, max_norm_y=0.5)
                    if pushing:
                        reward += weights.graveyard_push_bonus
                        log.append(f"GY_push(+{weights.graveyard_push_bonus:.1f},{len(pushing)}allies)")

        # ── Behind-giant support ──────────────────────────────────────────────
        if played_card != "giant":
            allies    = _troops_in_lane(tracks, "ally", played_lane)
            giants    = [t for t in allies if t.cls == "giant"]
            if giants:
                if played_card == "witch":
                    reward += weights.witch_behind_giant_bonus
                    log.append(f"WITCH_BEHIND_GIANT(+{weights.witch_behind_giant_bonus:.1f})")
                else:
                    supporting = [t for t in allies if t.cls != "giant"]
                    r = weights.behind_giant_bonus + len(supporting) * weights.behind_giant_per_troop
                    reward += r
                    log.append(f"behind_giant({played_card},{len(giants)}g,"
                                f"{len(supporting)}sup,+{r:.3f})")

    # ── Sparse: tower destroyed ───────────────────────────────────────────────
    for key in _ENEMY_TOWERS:
        if _just_destroyed(prev_health, curr_health, key):
            reward += weights.enemy_tower_bonus
            log.append(f"TOWER_DOWN({key},+{weights.enemy_tower_bonus:.1f})")
    for key in _ALLY_TOWERS:
        if _just_destroyed(prev_health, curr_health, key):
            reward += weights.ally_tower_penalty
            log.append(f"TOWER_LOST({key},{weights.ally_tower_penalty:.1f})")

    # ── Sparse: game over ─────────────────────────────────────────────────────
    if done:
        r = weights.win_bonus if won else weights.loss_penalty
        reward += r
        log.append("WIN" if won else "LOSS")

    if log:
        print(f"[reward] {reward:+.3f}  " + " | ".join(log))

    return float(reward)


# ── Summary string ────────────────────────────────────────────────────────────

def reward_summary(
    prev_health:            dict[str, float],
    curr_health:            dict[str, float],
    done:                   bool,
    won:                    bool,
    played_card:            str | None      = None,
    played_lane:            str | None      = None,
    played_y:               float | None    = None,
    spell_card:             str | None      = None,
    spell_lane:             str | None      = None,
    spell_disappeared_ids:  set             = None,
    spell_prev_map:         dict            = None,
    barrel_response_steps:  int | None      = None,
    elixir:                 float           = 5.0,
    tracks:                 list            = None,
    prev_track_ids:         set             = None,
    prev_tracks:            list            = None,
    hand:                   list[str]       = None,
    weights:                RewardWeights   = WEIGHTS,
) -> str:
    if tracks is None:              tracks = []
    if prev_tracks is None:         prev_tracks = []
    if hand is None:                hand = []
    if prev_track_ids is None:      prev_track_ids = set()
    if spell_disappeared_ids is None: spell_disappeared_ids = set()
    if spell_prev_map is None:      spell_prev_map = {}

    parts = []
    delta = _hp_delta(prev_health, curr_health)

    for key in _ENEMY_TOWERS:
        dmg = -delta[key]
        if dmg > 0:
            parts.append(f"dealt({key[6:]}:{dmg:.1f}%)")
    for key in _ALLY_TOWERS:
        dmg = -delta[key]
        if dmg > 0:
            parts.append(f"took({key[4:]}:{dmg:.1f}%)")
    for key in _ENEMY_TOWERS:
        if _just_destroyed(prev_health, curr_health, key):
            parts.append(f"TOWER_DOWN:{key}")
    for key in _ALLY_TOWERS:
        if _just_destroyed(prev_health, curr_health, key):
            parts.append(f"TOWER_LOST:{key}")

    # Princess kill
    current_ids = {getattr(t, "track_id", None) for t in tracks} - {None}
    disappeared = prev_track_ids - current_ids
    prev_map    = {getattr(t, "track_id", None): t for t in prev_tracks}
    prev_map.pop(None, None)
    for tid in disappeared:
        pt = prev_map.get(tid)
        if pt and pt.team == "enemy" and pt.cls == "princess":
            parts.append("PRINCESS_KILLED")

    if spell_card is not None:
        if len(spell_disappeared_ids) == 0:
            parts.append(f"SPELL_MISS({spell_card})")
        else:
            for tid in spell_disappeared_ids:
                pt = spell_prev_map.get(tid)
                if pt is None:
                    continue
                if spell_lane and not _in_lane(pt, spell_lane):
                    continue
                if pt.cls == "goblin_barrel":
                    _, norm_y = _troop_center(pt)
                    if norm_y > BARREL_DANGER_NORM_Y:
                        parts.append("BARREL_INTERCEPT")
                if pt.cls in _GOBLIN_CLASSES:
                    parts.append(f"spell_goblin({pt.cls})")
                if pt.cls == "princess" and spell_card == "arrows":
                    parts.append("SPELL_PRINCESS")

    if barrel_response_steps is not None:
        frac = max(0.0, 1.0 - (barrel_response_steps / weights.barrel_response_window))
        if frac > 0:
            parts.append(f"BARREL_RESPONSE(steps={barrel_response_steps})")

    if played_card is None:
        if _can_afford_any(hand, elixir):
            parts.append("idle")
    else:
        cost = CARD_COSTS.get(played_card, 3)
        if elixir < cost + ELIXIR_PATIENCE:
            parts.append(f"too_fast({played_card}@{elixir:.1f})")

    if played_card is not None and played_lane is not None:
        if played_card != "giant" and len(tracks) == 0:
            parts.append(f"nothing_on_screen({played_card})")
        if played_card != "giant":
            allies_this  = _ally_troops_in_lane(tracks, played_lane)
            allies_other = _ally_troops_in_lane(tracks, _other_lane(played_lane))
            if len(allies_this) == 0 and len(allies_other) > 0:
                parts.append(f"wrong_lane({played_card})")
        if played_card in _DEFENSIVE_UNITS:
            approaching = _enemy_approaching_in_lane(tracks, played_lane)
            if approaching:
                parts.append(f"def_purpose({played_card},{len(approaching)})")
            if played_y is not None:
                if _DEF_GOOD_Y_MIN <= played_y <= _DEF_GOOD_Y_MAX:
                    parts.append(f"def_good_pos(y={played_y:.0f})")
                elif played_y >= _DEF_BAD_Y_MIN:
                    parts.append(f"def_bad_pos(y={played_y:.0f})")
        if played_card == "graveyard":
            giants_ob = [
                t for t in _troops_in_lane(tracks, "ally", played_lane, max_norm_y=BRIDGE_NORM_Y)
                if t.cls == "giant"
            ]
            if giants_ob:
                parts.append(f"GY_BRIDGE({len(giants_ob)}g)")
            else:
                giants_lane = [t for t in _troops_in_lane(tracks, "ally", played_lane) if t.cls == "giant"]
                if giants_lane:
                    parts.append(f"GY_IN_PUSH({len(giants_lane)}g)")
                else:
                    pushing = _troops_in_lane(tracks, "ally", played_lane, max_norm_y=0.5)
                    if pushing:
                        parts.append(f"GY_push({len(pushing)})")
        if played_card != "giant":
            allies = _troops_in_lane(tracks, "ally", played_lane)
            giants = [t for t in allies if t.cls == "giant"]
            if giants:
                if played_card == "witch":
                    parts.append("WITCH_BEHIND_GIANT")
                else:
                    parts.append(f"behind_giant({played_card})")

    if done:
        parts.append("WIN" if won else "LOSS")

    return " | ".join(parts) if parts else "—"