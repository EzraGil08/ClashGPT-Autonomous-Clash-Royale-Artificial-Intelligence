"""
tracker.py — SORT-based multi-object tracker for Clash Royale.

Two independent track pools (ally / enemy) so IDs never cross teams.
Class-consistency check: an existing track will not be reassigned to a
detection whose class differs from the track's majority-voted class.

Public API
----------
tracker = CRTracker(max_age=8, min_hits=2, iou_threshold=0.25)
tracks  = tracker.update(detections)   # list[TrackResult]

TrackResult fields
------------------
  track_id  : int
  x1,y1,x2,y2 : float  (pixel coords in the cropped game region)
  cls       : str       (class label)
  conf      : float     (detection confidence, 0 if coasted)
  team      : str       ("ally" | "enemy")
  age       : int       (frames since first seen)
  hits      : int       (total matched frames)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: str
    team: str   # "ally" | "enemy"


@dataclass
class TrackResult:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    cls: str
    conf: float
    team: str
    age: int
    hits: int


# ── Kalman track ──────────────────────────────────────────────────────────────

_next_id = 1

def _new_id() -> int:
    global _next_id
    v = _next_id
    _next_id += 1
    return v


def _bbox_to_z(x1, y1, x2, y2):
    """[x_c, y_c, s, r]  s = area, r = aspect ratio (w/h)"""
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / float(h) if h != 0 else 1.0
    return np.array([[x], [y], [s], [r]], dtype=float)


def _z_to_bbox(x, y, s, r):
    """Inverse of _bbox_to_z."""
    w = np.sqrt(abs(s * r))
    h = abs(s) / w if w != 0 else 0
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


class Track:
    def __init__(self, det: Detection):
        self.id = _new_id()
        self.cls = det.cls
        self.team = det.team
        self.age = 1
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self._conf = det.conf

        # class vote history for consistency enforcement
        self._cls_votes: dict[str, int] = {det.cls: 1}

        # 7-state Kalman: [x_c, y_c, s, r, vx, vy, vs]
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)
        kf.R[2:,2:] *= 10.0
        kf.P[4:,4:] *= 1000.0
        kf.P       *= 10.0
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = _bbox_to_z(det.x1, det.y1, det.x2, det.y2)
        self.kf = kf

    # ------------------------------------------------------------------
    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, det: Detection):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self._conf = det.conf
        self._cls_votes[det.cls] = self._cls_votes.get(det.cls, 0) + 1
        # update majority class
        self.cls = max(self._cls_votes, key=self._cls_votes.__getitem__)
        self.kf.update(_bbox_to_z(det.x1, det.y1, det.x2, det.y2))

    def get_state(self):
        x, y, s, r = self.kf.x[:4].flatten()
        return _z_to_bbox(x, y, s, r)

    def to_result(self) -> TrackResult:
        x1, y1, x2, y2 = self.get_state()
        return TrackResult(
            track_id=self.id,
            x1=x1, y1=y1, x2=x2, y2=y2,
            cls=self.cls,
            conf=self._conf,
            team=self.team,
            age=self.age,
            hits=self.hits,
        )


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _match(tracks: list[Track], dets: list[Detection], iou_thresh: float):
    """
    Hungarian matching. Returns (matched_pairs, unmatched_dets, unmatched_tracks).
    Class-consistency: a det–track pair is forbidden if their classes differ
    (uses the track's majority class).
    """
    if not tracks or not dets:
        return [], list(range(len(dets))), list(range(len(tracks)))

    iou_matrix = np.zeros((len(tracks), len(dets)), dtype=float)
    for ti, t in enumerate(tracks):
        t_box = t.get_state()
        for di, d in enumerate(dets):
            # Class mismatch → zero IoU so Hungarian won't pair them
            if d.cls != t.cls:
                iou_matrix[ti, di] = 0.0
            else:
                iou_matrix[ti, di] = _iou(t_box, (d.x1,d.y1,d.x2,d.y2))

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched, unmatched_dets, unmatched_tracks = [], [], []
    matched_dets = set(); matched_tracks = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_thresh:
            matched.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)

    for di in range(len(dets)):
        if di not in matched_dets:
            unmatched_dets.append(di)
    for ti in range(len(tracks)):
        if ti not in matched_tracks:
            unmatched_tracks.append(ti)

    return matched, unmatched_dets, unmatched_tracks


# ── Public tracker ────────────────────────────────────────────────────────────

class CRTracker:
    """
    Parameters
    ----------
    max_age       : frames a track survives without a detection match
    min_hits      : detections needed before a track is returned to caller
    iou_threshold : minimum IoU to consider a det–track pair a match
    """

    def __init__(self, max_age: int = 4, min_hits: int = 2, iou_threshold: float = 0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._ally_tracks:  list[Track] = []
        self._enemy_tracks: list[Track] = []
        self.frame_count = 0

    def reset(self):
        global _next_id
        _next_id = 1
        self._ally_tracks = []
        self._enemy_tracks = []
        self.frame_count = 0

    def update(self, detections: list[Detection]) -> list[TrackResult]:
        """
        Feed one frame's detections → get confirmed tracks back.
        Detections with team="ally" and team="enemy" are processed in
        separate pools to prevent cross-team ID theft.
        """
        self.frame_count += 1

        ally_dets  = [d for d in detections if d.team == "ally"]
        enemy_dets = [d for d in detections if d.team == "enemy"]

        self._update_pool(self._ally_tracks,  ally_dets)
        self._update_pool(self._enemy_tracks, enemy_dets)

        # Prune dead tracks
        self._ally_tracks  = [t for t in self._ally_tracks  if t.time_since_update <= self.max_age]
        self._enemy_tracks = [t for t in self._enemy_tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self._ally_tracks + self._enemy_tracks:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append(t.to_result())
        return results

    def _update_pool(self, tracks: list[Track], dets: list[Detection]):
        # Predict all existing tracks
        for t in tracks:
            t.predict()

        matched, unmatched_dets, unmatched_tracks = _match(tracks, dets, self.iou_threshold)

        for ti, di in matched:
            tracks[ti].update(dets[di])

        # Kill tracks that got no match this frame (age handled inside predict)
        # New tracks from unmatched detections
        for di in unmatched_dets:
            tracks.append(Track(dets[di]))