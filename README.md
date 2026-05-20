# Clash Royale Autonomous Agent

A reinforcement learning agent that plays Clash Royale in real time on a Windows PC via an Android emulator. The agent uses computer vision (dual YOLOv11 models + SORT tracking) to perceive the game state, and a Maskable PPO policy to decide which cards to play and where to place them.

---

## Table of Contents

- [Overview](#overview)
- [Project Documents](#project-documents)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Large Assets (Google Drive)](#large-assets-google-drive)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Module Reference](#module-reference)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Training Details](#training-details)
- [Computer Vision Pipeline](#computer-vision-pipeline)
- [Known Limitations](#known-limitations)

---

## Overview

The agent runs a fixed 8-card deck against a standard opponent deck, playing entirely through screen capture and simulated mouse clicks — no game memory reading or API access. Every decision is made from pixels alone.

**Player deck:** Giant, Bowler, Witch, Graveyard, Guards, Arrows, Snowball, Minions

**Opponent deck (tracked):** Goblin Gang, Princess, Ice Spirit, Rocket, Goblin Barrel, Knight, Log, Cannon

The system operates in two modes:
- `train.py` — live training loop with MaskablePPO, checkpoint saving, TensorBoard logging, and a banner detector that pauses training when a game ends to collect win/loss labels.
- `debug_cv.py` — a live screen mirror with YOLO + SORT overlay for debugging the vision pipeline without the RL loop.

---

## Project Documents

| Document | Description |
|----------|-------------|
| [Report](report.pdf) | Full technical write-up |
| [Poster](poster.pdf) | Research poster |
| [Presentation](presentation.pdf) | Slide deck |

## Architecture

```
Screen (mss)
    │
    ▼
get_top_corner()          ← anchors all pixel coords to the game window
    │
    ├──► LBGG YOLO         ← detects troops & buildings
    ├──► SARG YOLO         ← detects spells & projectiles
    │         │
    │         ▼
    │     CRTracker (SORT) ← assigns stable IDs, separate ally/enemy pools
    │
    ├──► get_hand()        ← template matching → 4-card hand + next card
    ├──► get_elixir()      ← HSV color scan of elixir bar → float [0,10]
    └──► get_tower_health()← pixel scan of HP bars → 6 tower HP values
              │
              ▼
        build_observation() → 202-float vector
              │
              ▼
        MaskablePPO policy
              │
              ▼
        get_action_mask()   ← masks unaffordable / out-of-hand cards
              │
              ▼
        execute_action()    ← PyAutoGUI clicks (hand slot → placement)
```

---

## Repository Structure

```
Senior Research Project/
│
├── actions.py          # Action masking and PyAutoGUI execution
├── constants.py        # Card costs, placement coords, action-space layout
├── debug_cv.py         # Live vision debug window (no RL)
├── env.py              # Gymnasium CREnv — wraps the full pipeline
├── hand_detection.py   # Template-matching hand reader
├── misc_functions.py   # get_top_corner(), get_elixir()
├── observation.py      # build_observation() → 202-float vector
├── reward.py           # Hybrid dense/sparse reward function
├── tracker.py          # SORT-based multi-object tracker (CRTracker)
├── tower_health.py     # Pixel-scan HP reader for all 6 towers
├── train.py            # MaskablePPO training script
│
├── Images/             # Reference images used by CV modules (on Drive)
│   ├── cards_in_hand/  # 9 card template PNGs for get_hand()
│   ├── mini_cards/     # 8 mini-card templates for next-card detection
│   ├── test_images/    # Static game screenshots for offline testing
│   ├── banner.png      # End-of-game banner template for BannerDetector
│   └── elixir_blob.png # Legacy elixir reference (loaded by misc_functions)
│
├── YOLO/               # YOLO training data and weights (on Drive)
│   ├── LBGG/           # "Troops & Buildings" model
│   │   └── LBGG_model/train/weights/best.pt
│   └── SARG/           # "Spells & Ranged" model
│       └── SARG_model/train/weights/best.pt
│
└── Models/             # RL training outputs (on Drive)
    ├── checkpoints/    # Auto-saved .zip checkpoints every 512 steps
    ├── tensorboard/    # TensorBoard event files
    ├── monitor.csv     # SB3 Monitor wrapper episode log
    └── cr_agent_final.zip  # Final saved policy
```

---

## Large Assets (Google Drive)

The `Images/`, `YOLO/`, and `Models/` folders are too large for GitHub and are hosted on Google Drive:

**[Download Images, YOLO Weights & Models](https://drive.google.com/drive/folders/1zatBii3zCUWsibtnpse99O7PVHbQvDBs?usp=sharing)**

After downloading, place them at:
```
C:\Users\<you>\OneDrive\Desktop\Senior Research Project\
```
so the paths in `env.py`, `hand_detection.py`, `misc_functions.py`, and `train.py` resolve correctly. If you want to install elsewhere, do a find-and-replace on `BASE` in those files.

---

## Setup & Installation

**Requirements:** Windows 10/11, Python 3.10+, CUDA-capable GPU (training uses `device="cuda"`), an Android emulator (e.g. BlueStacks or LDPlayer) running Clash Royale.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
pip install stable-baselines3 sb3-contrib
pip install gymnasium
pip install opencv-python
pip install mss
pip install pyautogui
pip install filterpy
pip install scipy
pip install readchar
```

Or if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Training

```bash
# Start a fresh training run
python train.py

# Resume from the most recent checkpoint
python train.py --resume

# Resume from a specific checkpoint
python train.py --resume --checkpoint "Models/checkpoints/cr_model_1024_steps.zip"
```

**Calibration prompt** — at startup, the script asks you to mouse to the top-left corner of the game window (hold for 3 s), then the bottom-right corner (hold for 3 s). This sets the capture region. `get_top_corner()` then scans the captured frame for the characteristic dark-blue pixel at the game's top-left corner (BGR `[72, 30, 24]`, tolerance ±5) to anchor all subsequent coordinates.

**During training:**
- `S` — manually save a checkpoint
- `P` — pause / resume
- `Q` — save and quit cleanly

When the end-of-game banner is detected (template match score ≥ 0.80), training pauses and the terminal prompts for `y` (win) or `n` (loss) before asking you to start a new game and press Enter.

**TensorBoard:**
```bash
tensorboard --logdir "C:/Users/<you>/OneDrive/Desktop/Senior Research Project/Models/tensorboard"
```

### Debug Vision

```bash
python debug_cv.py
```

Opens a live OpenCV window (`CR Vision`) showing YOLO bounding boxes color-coded by team (green = ally, blue = enemy), track IDs, elixir, hand, next card, and tower HPs. Press `Q` to quit.

---

## Module Reference

### `constants.py`
Single source of truth for all card data and action-space geometry. Everything else imports from here — do not hardcode card names or placement coords elsewhere.

Key exports:
- `PLAYER_DECK` — ordered list of 8 cards the agent runs
- `CARD_COSTS` — elixir cost per card
- `CARD_PLACEMENTS` — dict mapping each card to its list of valid `(x, y)` drop coordinates, all relative to the `corner` anchor
- `HAND_SLOT_COORDS` — pixel centres of hand slots 0–3, corner-relative
- `ACTION_OFFSETS` — first action index for each card's placement block
- `WAIT_ACTION` — index 48 (the last action)
- `N_ACTIONS` — 49 total
- `ACTION_INDEX_MAP` — reverse map: `int → (card, placement_idx)` or `None` for WAIT

### `actions.py`
- `get_action_mask(hand, elixir) → np.ndarray(49, bool)` — masks any card not in the current 4-card hand or whose cost exceeds current elixir. WAIT is always unmasked.
- `execute_action(action_idx, hand, corner, capture_offset) → bool` — translates a flat action index into two PyAutoGUI clicks: first the hand slot, then the placement coordinate. Returns `False` for WAIT or unresolvable actions.
- `action_to_str(action_idx) → str` — human-readable label for logging (e.g. `giant@(242,756)`).

### `misc_functions.py`
- `get_top_corner(img) → (x, y)` — scans the captured frame pixel-by-pixel for BGR `[72, 30, 24]` (±5 tolerance), which is a fixed dark-blue pixel at the game's top-left edge. Returns the first match. This anchor is used as `corner` everywhere.
- `get_elixir(img, corner, last_elixir) → float` — reads the elixir bar by scanning 10 fixed x-positions at `y=978` (corner-relative) in HSV. Full bar detected via a single pixel check at `x=538`. Partial bars use fractional pixel counting. Upward spikes of more than 0.15 per frame are clamped to suppress color noise.

### `hand_detection.py`
- `get_hand(img, corner) → (hand: list[str], next_card: str)` — converts the frame to grayscale, crops the hand region (`y+820:y+951, x+124:x+541`), and runs `cv2.matchTemplate` (TM_CCOEFF_NORMED) against each of the 9 card templates. The top 4 spatially-separated matches (min 50px apart) are sorted left-to-right to give the 4-card hand. Next card is detected similarly against 8 mini-card templates in a separate crop (`y+930:y+986, x+29:x+75`).

### `tower_health.py`
- `get_tower_health(img, corner, last_health, tower_state) → (results, segments)` — scans 4 princess tower HP bars and 2 king tower HP bars by walking pixel columns and matching against known health bar BGR palettes (±18 tolerance). King bars are only scanned once a princess tower on that side reaches 0. Towers confirmed dead over `KING_CONFIRM_FRAMES=3` consecutive frames are permanently locked at 0 in `tower_state["dead"]` to prevent resurrection artifacts.
- `make_tower_state() → dict` — creates a fresh per-episode state dict with `dead: set` and `king_confirm` counters.
- `debug_draw(image, corner, last_health, tower_state, out_path)` — saves an annotated PNG showing every scanned pixel color-coded (green = health, orange = bar end, yellow = unexpected/king polled).

### `tracker.py`
- `CRTracker(max_age, min_hits, iou_threshold)` — maintains two independent SORT track pools (ally and enemy) so cross-team ID collisions are impossible. Class-consistency enforcement: a detection is forbidden from matching a track whose majority-voted class differs.
- `tracker.update(detections) → list[TrackResult]` — runs Kalman predict → Hungarian assignment → update. Tracks returned only after `min_hits=2` confirmed frames. Tracks survive up to `max_age=8` frames without a match.
- `TrackResult` fields: `track_id, x1, y1, x2, y2, cls, conf, team, age, hits`

### `observation.py`
- `build_observation(tracks, hand, next_card, elixir, tower_health) → np.ndarray(202,)` — assembles the flat observation vector. See [Observation Space](#observation-space) below.

### `reward.py`
- `compute_reward(...) → float` — evaluates the hybrid dense/sparse reward. See [Reward Function](#reward-function) below.
- `reward_summary(...)  → str` — human-readable string of which reward components fired, used in the render overlay.

### `env.py`
- `CREnv(calibration_region, render_mode)` — the Gymnasium environment. Loads both YOLO models on init. Runs YOLO every `YOLO_SKIP_FRAMES=3` steps (every other 2 frames reuse the last tracker state to reduce inference latency). HP noise filter suppresses tower deltas < 0.5% between steps. Spell-miss detection checks whether Arrows or Snowball caused a tower HP drop or troop count drop; if neither, fires the `spell_missed` penalty. Episode terminates when all 3 towers on either side reach 0, or when `force_done()` is called by the banner handler.
- `render_mode="human"` — opens a `CR Agent` OpenCV window with bounding boxes, HUD overlay, and per-step reward.

### `train.py`
- `BannerDetector` — polls the capture region for `banner.png` via template matching every step (with a 300-step cooldown after a detection) to identify game-end screens.
- `EpisodeStatsCallback` — logs per-episode reward, length, and tower HPs to TensorBoard. Handles the banner-detected pause flow.
- `EntropyCoefficientCallback` — linearly decays `model.ent_coef` from 0.05 → 0.01 over the full training run (done manually per step because MaskablePPO does not accept a callable for `ent_coef`).
- `PauseAndSaveCallback` — background keyboard listener thread for S/P/Q hotkeys.

---

## Observation Space

`Box(low=0.0, high=1.0, shape=(202,), dtype=float32)`

| Indices | Content | Size |
|---------|---------|------|
| 0 – 149 | Up to 30 tracked troops × 5 features | 150 |
| 150 – 185 | 4 hand slots × 9-class one-hot (CARD_NAMES) | 36 |
| 186 – 194 | Next card one-hot (9 classes) | 9 |
| 195 | Elixir ÷ 10 | 1 |
| 196 – 199 | Princess tower HPs ÷ 100 (ally_left, ally_right, enemy_left, enemy_right) | 4 |
| 200 | Ally king HP ÷ 100 | 1 |
| 201 | Enemy king HP ÷ 100 | 1 |

**Troop feature vector (5 floats):**

| Index | Feature | Range |
|-------|---------|-------|
| 0 | Bounding box centre x ÷ game width | [0, 1] |
| 1 | Bounding box centre y ÷ game height | [0, 1] |
| 2 | Team (1.0 = ally, 0.0 = enemy) | {0, 1} |
| 3 | Class index ÷ (n\_classes − 1) | [0, 1] |
| 4 | min(track age, 60) ÷ 60 | [0, 1] |

If more than 30 troops are tracked, skeletons are dropped first (oldest first), then non-skeletons are trimmed by age.

**Troop classes (18 total):** giant, bowler, witch, graveyard, guard, arrows, snowball, minion, princess, ice\_spirit, rocket, goblin\_barrel, knight, log, cannon, skeleton, goblin, spear\_goblin

**Card classes (9, includes "empty"):** arrows, bowler, giant, graveyard, guards, minions, snowball, witch, empty

Team assignment for spawned units (skeleton, goblin, spear\_goblin) uses position: `norm_y > 0.5` = ally (bottom half of arena). All other units use the `e_` prefix convention from the YOLO label names.

---

## Action Space

`Discrete(49)` — one action per valid placement coordinate per card, plus one WAIT action.

| Card | # Placements | Action Indices |
|------|-------------|----------------|
| giant | 6 | 0 – 5 |
| bowler | 6 | 6 – 11 |
| witch | 5 | 12 – 16 |
| graveyard | 2 | 17 – 18 |
| guards | 6 | 19 – 24 |
| arrows | 8 | 25 – 32 |
| snowball | 8 | 33 – 40 |
| minions | 6 | 41 – 46 |
| WAIT | — | 48 |

Lane assignment (used by reward): placement `x < 280` = left lane, `x ≥ 280` = right lane.

The action mask zeros out any card not currently in the 4-card hand or whose elixir cost exceeds current elixir. WAIT (index 48) is always unmasked. The mask is fed to MaskablePPO via the `ActionMasker` wrapper so the policy never samples an illegal action.

---

## Reward Function

Hybrid dense + sparse design. All weights live in `RewardWeights` in `reward.py` and can be tuned without touching logic.

### Dense (every step)

| Signal | Weight | Notes |
|--------|--------|-------|
| Enemy tower HP lost (per 1%) | +0.01 | Encourages attacking |
| Ally tower HP lost (per 1%) | −0.02 | Penalises taking damage (higher weight than offense) |
| Ally troop in enemy half — depth bonus | +0.003 × depth | Positional pressure per troop per step |
| Enemy troop in ally half — depth penalty | −0.003 × depth | Discourages letting troops through |

Depth is `0.5 − norm_y` for ally troops in the enemy half, and `norm_y − 0.5` for enemy troops in the ally half, where `norm_y=0` is the top of the arena (enemy side).

### Sparse (event-triggered)

| Event | Reward |
|-------|--------|
| Enemy tower destroyed | +2.0 |
| Ally tower destroyed | −2.0 |
| Win | +5.0 |
| Loss | −5.0 |
| Spell (Arrows/Snowball) missed | −0.3 |
| Card played at elixir ≥ 7 | +0.05 (efficiency bonus) |
| Card played at elixir < 3 | −0.15 (spam penalty) |
| WAIT when elixir ≥ 8 and cards available | −0.05 (hoarding penalty) |
| WAIT when can't afford anything and elixir < 4 | +0.02 (saving bonus) |
| Any non-giant played into a lane with a friendly giant | +0.4 base + 0.1 per additional ally in that lane |
| Graveyard played with allies already in enemy half of that lane | +1.5 |

The push bonuses are designed to teach the agent the core Giant–Graveyard combo: place the Giant first, let it advance, then drop the Graveyard when the Giant is in the enemy half.

---

## Training Details

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Algorithm | MaskablePPO (sb3-contrib) | Handles discrete action masking natively |
| `n_steps` | 512 | One gradient update per ~10–15 min of real play (~2–3 s/step) |
| `batch_size` | 128 | 4 minibatches per rollout for clean gradients |
| `n_epochs` | 10 | |
| `gamma` | 0.97 | ~33-step effective horizon; matches card→tower-damage delay |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `ent_coef` | 0.05 → 0.01 (linear) | High early entropy prevents wait-spam collapse |
| `learning_rate` | 3e-4 → 1e-5 (linear) | |
| `vf_coef` | 0.5 | |
| `max_grad_norm` | 0.5 | |
| `device` | cuda | |
| Total timesteps | 1,000,000 | |
| Checkpoint frequency | every 512 steps | Saved to `Models/checkpoints/` |

Checkpoints are named `cr_model_<N>_steps.zip`. The resume flag (`--resume`) auto-detects the most recent checkpoint by file modification time and extracts the step count from the filename to correctly offset `total_timesteps`.

---

## Computer Vision Pipeline

### Game Window Anchoring
All pixel coordinates throughout the codebase are stored as offsets from a `corner = (x, y)` anchor. `get_top_corner()` finds this anchor by scanning for the fixed dark-blue border pixel at the top-left of the game UI (BGR `[72, 30, 24]`, ±5). This means the code works regardless of where the emulator window is positioned on screen.

### YOLO Models
Two YOLOv11 models run in parallel on the cropped game region (420×1000 px), with results merged before the tracker:
- **LBGG** (`best.pt` in `YOLO/LBGG/`) — detects troops and buildings
- **SARG** (`best.pt` in `YOLO/SARG/`) — detects spells and projectiles

Inference threshold: `conf=0.35`, `iou=0.45`. YOLO runs every 3rd frame (`YOLO_SKIP_FRAMES=3`); on skipped frames the tracker coasts on its Kalman predictions.

### SORT Tracker
Two independent SORT pools (ally/enemy) with a class-consistency check: detections can only match tracks of the same class. Parameters: `max_age=8`, `min_hits=2`, `iou_threshold=0.25`. The 7-state Kalman filter tracks `[x_c, y_c, area, aspect_ratio, vx, vy, v_area]`.

### Hand Detection
Template matching (TM_CCOEFF_NORMED) against 9 reference PNGs in `Images/cards_in_hand/`. The top 4 matches with centres at least 50 px apart (left-to-right order) identify the current hand. Next card uses 8 mini-card templates in `Images/mini_cards/`.

### Elixir Reading
10 fixed pixel positions along the elixir bar are sampled in HSV. Each full-bar pixel scores +1 elixir; partial-bar pixels use sub-pixel interpolation over a 39-pixel scan window to give fractional elixir values. Upward spikes > 0.15 per frame are clamped.

### Tower Health Reading
HP bars are scanned column-by-column in BGR, matching against known palette colors (±18 tolerance). Princess bars stop scanning on a "terminate" color (the empty bar edge). King bars are only activated once a princess tower on that side falls. A 3-frame confirmation window prevents single noisy frames from declaring towers dead prematurely.

---

## Known Limitations

- **Windows only** — uses `mss` for screen capture, `pyautogui` for input, and `msvcrt` for keyboard hotkeys.
- **Fixed emulator resolution** — all pixel coordinates assume the emulator is running at the specific resolution captured during development. Rescaling the window will break hand detection, elixir reading, and tower health scanning.
- **Manual game reset** — the agent does not navigate Clash Royale menus. A human must start each game and press Enter in the terminal when the game is ready.
- **No opponent modeling** — the opponent's deck is tracked by YOLO but not explicitly used in strategy beyond positional awareness.
- **Single environment** — training runs in a single non-vectorized environment, so sample efficiency is limited by real-time game speed (~2–3 s/step).
