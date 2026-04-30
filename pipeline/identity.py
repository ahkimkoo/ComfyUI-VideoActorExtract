"""Face recognition and identity clustering module.

Uses InsightFace for face embedding extraction and cosine similarity
clustering to merge tracks belonging to the same actor.

Key constraint: tracks that co-occur (overlap in time) can NEVER be merged.
This prevents merging different people who happen to have similar faces.
"""

import math
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

from pipeline.tracker import FrameRecord
from pipeline.detector import BoundingBox
from core.config import DEFAULT_FACE_THRESHOLD, DEFAULT_MIN_FACE_CONFIDENCE


class IdentityCluster:
    """Identity clustering using InsightFace face embeddings."""

    def __init__(self, threshold: float = DEFAULT_FACE_THRESHOLD, model_dir: str = ""):
        self.threshold = threshold
        self.model = None
        self._loaded = False
        self.model_dir = model_dir
        self._face_bbox_areas: Dict[int, Dict[int, int]] = {}
        # {track_id: {frame_idx: face_bbox_area}}

    def _ensure_loaded(self):
        """Lazy-load InsightFace model."""
        if self._loaded:
            return
        try:
            from insightface.app import FaceAnalysis

            # Set proxy for model download
            proxy = os.environ.get("http_proxy", os.environ.get("HTTP_PROXY", ""))
            if proxy:
                os.environ.setdefault("HTTP_PROXY", proxy)
                os.environ.setdefault("HTTPS_PROXY", proxy)

            # Resolve model directory: caller-specified > ComfyUI models dir > default
            model_dir = self.model_dir
            if not model_dir or not os.path.isdir(os.path.join(model_dir, "buffalo_l")):
                # Try ComfyUI folder_paths "video-actor-extract"
                try:
                    import folder_paths

                    paths = folder_paths.get_folder_paths("video-actor-extract")
                    for p in paths:
                        candidate = os.path.join(p, "buffalo_l")
                        if os.path.isdir(candidate):
                            model_dir = p
                            break
                except Exception:
                    pass

            if not model_dir or not os.path.isdir(os.path.join(model_dir, "buffalo_l")):
                # Fallback to default InsightFace location
                model_dir = os.path.expanduser("~/.insightface/models")

            # Check if model files exist before attempting to load
            # buffalo_l needs det_10g.onnx and w600k_r50.onnx
            required_files = [
                "1k3d68.onnx",
                "2d106det.onnx",
                "det_10g.onnx",
                "genderage.onnx",
                "w600k_r50.onnx",
            ]
            model_path = os.path.join(model_dir, "buffalo_l")
            has_all = all(
                os.path.exists(os.path.join(model_path, f)) for f in required_files
            )

            if not has_all:
                print(
                    "[Identity] InsightFace buffalo_l model not found locally, skipping face clustering"
                )
                print(
                    "[Identity] Run: python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)\""
                )
                print("[Identity] to download the model (~350MB)")
                self._loaded = False
                self.model = None
                return

            self.model = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                root=model_dir,
            )
            self.model.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU
            self._loaded = True
            print("[Identity] InsightFace loaded successfully")
        except Exception as e:
            print(f"[Identity] Failed to load InsightFace: {e}")
            print("[Identity] Will use track-only mode (no face clustering)")
            self._loaded = False
            self.model = None

    def _get_face_embedding(
        self, frame: np.ndarray, bbox: BoundingBox
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Extract face embedding from a cropped region of a frame.
        Tries multiple crop sizes to maximize face detection chances.

        Args:
            frame: Full BGR frame
            bbox: Bounding box of the person

        Returns:
            Tuple of (face_embedding, face_bbox_area) where both are None
            if no face detected. face_bbox_area is (x2-x1)*(y2-y1) in pixels.
        """
        self._ensure_loaded()
        if self.model is None:
            return None, None

        h, w = frame.shape[:2]

        # Try multiple expansion values to maximize face detection
        # Start with moderate expansion, then try larger, then full frame
        expand_values = [0.6, 1.0, 1.5, 2.0]

        for expand in expand_values:
            bw = bbox.width * expand
            bh = bbox.height * expand
            x1 = max(0, int(bbox.x1 - bw))
            y1 = max(0, int(bbox.y1 - bh))
            x2 = min(w, int(bbox.x2 + bw))
            y2 = min(h, int(bbox.y2 + bh))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            faces = self.model.get(crop)
            if not faces:
                continue

            # Return the embedding of the face with highest detection confidence
            best_face = max(faces, key=lambda f: f.det_score)
            if best_face.det_score < DEFAULT_MIN_FACE_CONFIDENCE:
                continue

            face_area = int(
                (best_face.bbox[2] - best_face.bbox[0])
                * (best_face.bbox[3] - best_face.bbox[1])
            )
            return best_face.embedding, face_area

        # Fallback: try the full frame (for cases where bbox is very inaccurate)
        faces = self.model.get(frame)
        if faces:
            best_face = max(faces, key=lambda f: f.det_score)
            if best_face.det_score >= DEFAULT_MIN_FACE_CONFIDENCE:
                face_area = int(
                    (best_face.bbox[2] - best_face.bbox[0])
                    * (best_face.bbox[3] - best_face.bbox[1])
                )
                return best_face.embedding, face_area

        return None, None

    def _get_track_embedding_with_count(self, track_records, frames):
        """Get aggregated face embedding for a track.

        Returns:
            (embedding, face_count, face_areas) where face_areas maps
            frame_idx -> face_bbox_area in pixels.
        """
        embeddings = []
        n = len(track_records)
        if n == 0:
            return None, 0, {}

        # Sort records by mask area (descending) to prioritize frames where person is largest
        sorted_records = sorted(
            track_records,
            key=lambda r: (r.x2 - r.x1) * (r.y2 - r.y1),
            reverse=True,
        )

        # Sample up to 30 frames, prioritizing largest area frames
        max_samples = 30
        if n <= max_samples:
            sample_records = sorted_records
        else:
            # Take top frames by area, but also ensure temporal coverage
            top_by_area = sorted_records[:20]

            # Also sample evenly from the original (temporal) order
            step = n / 10.0
            temporal_indices = [int(i * step) for i in range(10)]
            top_temporal = [track_records[i] for i in temporal_indices]

            # Combine and deduplicate by frame_idx
            seen_frames = set()
            sample_records = []
            for rec in top_by_area + top_temporal:
                if rec.frame_idx not in seen_frames:
                    seen_frames.add(rec.frame_idx)
                    sample_records.append(rec)

        face_count = 0
        face_areas: Dict[int, int] = {}
        for rec in sample_records:
            frame = frames.get(rec.frame_idx, None)
            if frame is None:
                continue
            bbox = BoundingBox(rec.x1, rec.y1, rec.x2, rec.y2)
            emb, face_area = self._get_face_embedding(frame, bbox)
            if emb is not None:
                embeddings.append(emb)
                face_count += 1
                if face_area is not None:
                    face_areas[rec.frame_idx] = face_area

        if not embeddings:
            return None, face_count, face_areas

        # Simple average of all successful embeddings
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
        return avg_emb, face_count, face_areas

    def cluster_tracks(
        self,
        track_records: Dict[int, List[FrameRecord]],
        frames: Dict[int, np.ndarray],
        min_track_length: int = 5,
        max_lost_frames: int = 30,
    ) -> Dict[int, str]:
        """
        Cluster tracks into actor identities.

        Key constraint: tracks that co-occur (overlap in time) can NEVER be merged.
        This prevents merging different people who happen to have similar faces.

        Args:
            track_records: {track_id: [FrameRecord, ...]}
            frames: {frame_idx: numpy_frame} - all frames needed for face extraction
            min_track_length: Minimum number of frames for a track to be considered
            max_lost_frames: Tracker lost-frame tolerance. When the temporal gap
                between a track and an actor's member is below this value, the
                tracker had the opportunity to merge but chose not to — so face
                similarity is penalised to respect the tracker's decision.

        Returns:
            {track_id: "actor_0", ...}
        """
        self._ensure_loaded()

        # Reset face bbox areas for this clustering run
        self._face_bbox_areas = {}

        # Filter out short tracks
        valid_tracks = {
            tid: recs
            for tid, recs in track_records.items()
            if len(recs) >= min_track_length
        }

        # Note: track splitting is now handled by the caller (actor_extractor.py)
        # before calling cluster_tracks(). Each key in valid_tracks is already
        # a clean single-person subtrack.

        if not valid_tracks:
            return {}

        # If no face model, assign each track a unique actor ID
        if self.model is None:
            return {tid: f"actor_{i}" for i, tid in enumerate(valid_tracks.keys())}

        # Extract face embeddings for each track
        print("[Identity] Extracting face embeddings...")
        track_embeddings: Dict[int, Optional[np.ndarray]] = {}
        face_detection_counts: Dict[int, int] = {}

        for tid, recs in valid_tracks.items():
            emb, count, face_areas = self._get_track_embedding_with_count(recs, frames)
            track_embeddings[tid] = emb
            face_detection_counts[tid] = count
            if face_areas:
                self._face_bbox_areas[tid] = face_areas

        # Debug: print face detection stats
        tracks_with_faces = sum(1 for e in track_embeddings.values() if e is not None)
        print(
            f"[Identity] Face detection: {tracks_with_faces}/{len(valid_tracks)} tracks have face embeddings"
        )
        for tid, count in face_detection_counts.items():
            if count > 0:
                print(f"[Identity]   Track {tid}: {count} faces detected")

        # Compute time ranges for each track
        track_time_ranges: Dict[int, Tuple[int, int]] = {}
        for tid, recs in valid_tracks.items():
            frame_indices = [r.frame_idx for r in recs]
            track_time_ranges[tid] = (min(frame_indices), max(frame_indices))

        # Check if two tracks co-occur (overlap in time)
        def tracks_cooccur(tid_a, tid_b) -> bool:
            """Two tracks co-occur if their time ranges overlap."""
            lo_a, hi_a = track_time_ranges[tid_a]
            lo_b, hi_b = track_time_ranges[tid_b]
            return lo_a <= hi_b and lo_b <= hi_a

        # Use greedy clustering with co-occurrence constraint
        # Process tracks in order. For each track, find the best matching actor.
        # An actor can accept a new track ONLY if no existing member co-occurs with it.

        actor_id_counter = 0
        track_to_actor: Dict[int, str] = {}
        # actor_id -> list of track IDs belonging to this actor
        actor_tracks: Dict[str, List[int]] = {}

        # Sort tracks: those with embeddings first
        sorted_tracks = sorted(
            valid_tracks.keys(),
            key=lambda tid: 0 if track_embeddings[tid] is not None else 1,
        )

        # Margin for spatial tiebreaker: when top candidates' adjusted
        # similarity scores differ by ≤ this value, use spatial centroid
        # distance to break the tie.  Prevents incorrect merges between
        # people at different positions who happen to have similar face
        # embeddings (e.g. child vs small adult).
        SIMILARITY_TIE_MARGIN = 0.1

        for tid in sorted_tracks:
            emb = track_embeddings[tid]
            if emb is None:
                # No face detected, will handle later
                continue

            # Collect all viable merge candidates with their scores
            candidates: List[Tuple[str, float]] = []  # (actor_id, adjusted_sim)

            for actor_id, member_tids in actor_tracks.items():
                # Check co-occurrence: this track must NOT co-occur with ANY member
                can_merge = True
                for member_tid in member_tids:
                    if tracks_cooccur(tid, member_tid):
                        can_merge = False
                        break

                if not can_merge:
                    continue

                # Compute average similarity to all members
                member_embs = [
                    track_embeddings[mt]
                    for mt in member_tids
                    if track_embeddings[mt] is not None
                ]
                if not member_embs:
                    continue
                sims = [float(np.dot(emb, me)) for me in member_embs]
                avg_sim = np.mean(sims)

                # Compute minimum temporal gap to any actor member
                min_gap = float("inf")
                for member_tid in member_tids:
                    m_lo, m_hi = track_time_ranges[member_tid]
                    t_lo, t_hi = track_time_ranges[tid]
                    gap = max(0, max(m_lo, t_lo) - min(m_hi, t_hi))
                    min_gap = min(min_gap, gap)

                # Temporal gap penalty: if gap < max_lost_frames, the tracker had
                # the opportunity to merge but chose not to → penalize face similarity.
                # Exception: very small gaps (≤ 2 frames) are usually tracker timing
                # artifacts (mask centroid shifted for 1-2 frames due to motion),
                # not deliberate separation decisions — skip penalty for those.
                TRACKER_GRACE_GAP = 2

                if max_lost_frames > 0 and min_gap < max_lost_frames:
                    if min_gap <= TRACKER_GRACE_GAP:
                        adjusted_sim = avg_sim
                    else:
                        penalty = min_gap / max_lost_frames
                        adjusted_sim = avg_sim * penalty
                else:
                    adjusted_sim = avg_sim

                print(
                    f"[Identity]   Track {tid} vs {actor_id}: "
                    f"raw_sim={avg_sim:.3f}, gap={min_gap:.0f}, "
                    f"adjusted_sim={adjusted_sim:.3f}"
                )

                # Temporal boost: for large-gap non-blocked tracks, use
                # temporal evidence to complement unreliable face similarity.
                # When no other track appears between two non-adjacent tracks,
                # it's more likely the same person left and returned.
                if max_lost_frames > 0 and min_gap >= max_lost_frames:
                    has_blocker = self._has_temporal_blocker(
                        tid,
                        member_tids,
                        track_time_ranges,
                        set(valid_tracks.keys()),
                        track_records,
                    )
                    temporal_conf = self._compute_temporal_confidence(
                        min_gap,
                        max_lost_frames,
                        has_blocker,
                    )
                    if temporal_conf >= 0.5 and adjusted_sim >= 0.1:
                        merge_score = max(adjusted_sim, temporal_conf)
                        print(
                            f"[Identity]   Track {tid} vs {actor_id}: "
                            f"temporal_boost(conf={temporal_conf:.3f}, "
                            f"blocker={has_blocker}) -> "
                            f"merge_score={merge_score:.3f}"
                        )
                        adjusted_sim = merge_score

                candidates.append((actor_id, adjusted_sim))

            if not candidates:
                # No viable merge target — create new actor
                actor_id = f"actor_{actor_id_counter}"
                actor_id_counter += 1
                track_to_actor[tid] = actor_id
                actor_tracks[actor_id] = [tid]

                lo_tid, hi_tid = track_time_ranges[tid]
                print(
                    f"[Identity] New actor {actor_id}: track {tid}({lo_tid}-{hi_tid})"
                )
                continue

            # Select best candidate: sort by adjusted similarity descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_actor, best_similarity = candidates[0]

            # Spatial proximity tiebreaker: when face similarities are close,
            # prefer merging with the actor whose members are spatially closer.
            # This prevents incorrect merges between people at different
            # positions who happen to have similar face embeddings
            # (e.g. child vs small adult).
            if len(candidates) >= 2:
                top_sim = candidates[0][1]
                competitive = [
                    (aid, sim)
                    for aid, sim in candidates
                    if top_sim - sim <= SIMILARITY_TIE_MARGIN
                ]
                if len(competitive) >= 2:
                    tid_centroid = self._track_median_centroid(
                        valid_tracks.get(tid, [])
                    )
                    if tid_centroid is not None:
                        best_dist = float("inf")
                        tie_winner = None
                        for aid, sim in competitive:
                            # Compute min distance to any member of this actor
                            for mt in actor_tracks[aid]:
                                mt_centroid = self._track_median_centroid(
                                    valid_tracks.get(mt, [])
                                )
                                if mt_centroid is not None:
                                    d = (
                                        (tid_centroid[0] - mt_centroid[0]) ** 2
                                        + (tid_centroid[1] - mt_centroid[1]) ** 2
                                    ) ** 0.5
                                    if d < best_dist:
                                        best_dist = d
                                        tie_winner = aid

                        if tie_winner is not None and tie_winner != best_actor:
                            print(
                                f"[Identity]   Track {tid}: spatial tiebreaker "
                                f"(top_sim={top_sim:.3f}, "
                                f"margin={SIMILARITY_TIE_MARGIN}) -> "
                                f"chose {tie_winner} (dist={best_dist:.0f}px) "
                                f"over {best_actor} (sim={best_similarity:.3f})"
                            )
                            best_actor = tie_winner

            if best_similarity >= self.threshold:
                # Merge into existing actor
                track_to_actor[tid] = best_actor
                actor_tracks[best_actor].append(tid)

                lo_tid, hi_tid = track_time_ranges[tid]
                print(
                    f"[Identity] Merged track {tid}({lo_tid}-{hi_tid}) -> "
                    f"{best_actor} (adjusted_sim={best_similarity:.3f})"
                )
            else:
                # Below threshold — create new actor
                actor_id = f"actor_{actor_id_counter}"
                actor_id_counter += 1
                track_to_actor[tid] = actor_id
                actor_tracks[actor_id] = [tid]

                lo_tid, hi_tid = track_time_ranges[tid]
                print(
                    f"[Identity] New actor {actor_id}: track {tid}({lo_tid}-{hi_tid})"
                )

        # Assign unique IDs to tracks without face embeddings
        faceless_tracks = [tid for tid, emb in track_embeddings.items() if emb is None]
        if faceless_tracks:
            self._merge_faceless_by_overlap(
                faceless_tracks,
                valid_tracks,
                track_to_actor,
                track_embeddings,
                actor_id_counter,
            )
            # Update actor_id_counter after faceless merge
            all_actors = set(track_to_actor.values())
            actor_id_counter = (
                max(int(a.split("_")[1]) for a in all_actors if a.startswith("actor_"))
                + 1
                if all_actors
                else 0
            )

        # Renumber actor IDs sequentially
        unique_actors = sorted(set(track_to_actor.values()))
        actor_map = {aid: idx for idx, aid in enumerate(unique_actors)}
        track_to_actor = {
            tid: f"actor_{actor_map[old_name]}"
            for tid, old_name in track_to_actor.items()
        }

        print(f"[Identity] Final: {len(set(track_to_actor.values()))} unique actors")

        return track_to_actor

    def get_face_bbox_areas(self) -> Dict[int, Dict[int, int]]:
        """Return face bbox areas detected during clustering.

        Returns:
            {track_id: {frame_idx: face_bbox_area_pixels}}
        """
        return self._face_bbox_areas

    def _merge_faceless_by_overlap(
        self,
        faceless_tracks: list,
        valid_tracks: dict,
        track_to_actor: dict,
        track_embeddings: dict,
        actor_id_counter: int,
    ):
        """Try to assign faceless tracks to existing actors by spatial-temporal overlap."""
        # Build frame-range per existing actor
        actor_frame_ranges: Dict[str, Tuple[int, int]] = {}
        actor_bboxes: Dict[str, List[BoundingBox]] = {}
        for tid, aid in track_to_actor.items():
            if aid not in actor_frame_ranges:
                actor_frame_ranges[aid] = (float("inf"), float("-inf"))
                actor_bboxes[aid] = []
            recs = valid_tracks[tid]
            frame_indices = [r.frame_idx for r in recs]
            lo, hi = actor_frame_ranges[aid]
            actor_frame_ranges[aid] = (
                min(lo, min(frame_indices)),
                max(hi, max(frame_indices)),
            )
            for r in recs:
                actor_bboxes[aid].append(BoundingBox(r.x1, r.y1, r.x2, r.y2))

        for tid in faceless_tracks:
            recs = valid_tracks[tid]
            track_frames = set(r.frame_idx for r in recs)
            track_lo, track_hi = min(track_frames), max(track_frames)

            best_actor = None
            best_score = 0.0

            for aid, (a_lo, a_hi) in actor_frame_ranges.items():
                # Time range overlap
                overlap_start = max(track_lo, a_lo)
                overlap_end = min(track_hi, a_hi)
                if overlap_start > overlap_end:
                    continue
                time_overlap = (overlap_end - overlap_start + 1) / max(
                    1, track_hi - track_lo + 1
                )
                if time_overlap < 0.3:
                    continue

                # Bbox overlap (use median bbox of each)
                track_bboxes = [BoundingBox(r.x1, r.y1, r.x2, r.y2) for r in recs]
                med_track = self._median_bbox(track_bboxes)
                med_actor = self._median_bbox(actor_bboxes[aid])
                iou = self._bbox_iou(med_track, med_actor)

                score = time_overlap * 0.5 + iou * 0.5
                if score > best_score:
                    best_score = score
                    best_actor = aid

            if best_actor is not None and best_score > 0.3:
                track_to_actor[tid] = best_actor
                print(
                    f"[Identity] Merged faceless track {tid} -> {best_actor} (score={best_score:.2f})"
                )
            else:
                # Assign a new unique ID
                new_id = f"actor_{actor_id_counter}"
                actor_id_counter += 1
                track_to_actor[tid] = new_id
                print(f"[Identity] Faceless track {tid} unmatched, assigned {new_id}")

    @staticmethod
    def _split_mixed_tracks(
        track_records: Dict[int, List[FrameRecord]],
        min_subtrack_length: int = 5,
    ) -> Tuple[Dict[int, List[FrameRecord]], Dict[int, int]]:
        """Split tracks that contain content shifts (different people).

        Uses a dual-signal dominant outlier detection algorithm on centroid
        displacement AND bbox area ratio to detect points where the tracked
        person changes identity. Area ratio is the primary signal (different
        person = different body size); displacement is secondary.

        Args:
            track_records: {track_id: [FrameRecord, ...]}
            min_subtrack_length: Minimum frames for a subtrack to be kept.

        Returns:
            Tuple of:
            - New track_records with mixed tracks split into sub-tracks.
            - Mapping from new track IDs to original parent track IDs.
        """
        new_records = {}
        new_to_original = {}
        next_id = 0

        for tid, records in track_records.items():
            if len(records) < min_subtrack_length * 2:
                # Too short to split meaningfully
                new_records[next_id] = records
                new_to_original[next_id] = tid
                next_id += 1
                continue

            # Recursively split this track
            subtracks = IdentityCluster._recursive_split(records, min_subtrack_length)

            for sub in subtracks:
                new_records[next_id] = sub
                new_to_original[next_id] = tid
                next_id += 1

        return new_records, new_to_original

    @staticmethod
    def _recursive_split(
        records: List[FrameRecord],
        min_subtrack_length: int,
    ) -> List[List[FrameRecord]]:
        """Recursively find and apply splits using dual-signal IQR outlier detection.

        Computes per-frame centroid displacement and bbox area ratio. A frame
        is a split candidate only when BOTH signals show IQR outliers
        simultaneously, which distinguishes genuine identity changes from
        normal motion or jitter.

        Args:
            records: Sequential FrameRecord list for one track.
            min_subtrack_length: Minimum frames for each resulting subtrack.

        Returns:
            List of subtrack record lists (at least one, always contains all input).
        """
        if len(records) < min_subtrack_length * 2:
            return [records]

        n = len(records)
        if n < 3:
            return [records]

        # ── 1. Compute per-frame signals ──────────────────────────────
        # centroid (cx, cy) and area (w*h) for each record
        centroids = []
        areas = []
        for r in records:
            cx = (r.x2 + r.x1) / 2.0
            cy = (r.y2 + r.y1) / 2.0
            centroids.append((cx, cy))
            areas.append(max((r.x2 - r.x1) * (r.y2 - r.y1), 1e-6))

        # ── 2. Frame-to-frame signals ────────────────────────────────
        displacements = []  # |centroid_i - centroid_{i-1}|
        area_ratios = []  # area_i / area_{i-1}  (always >= 1)
        for i in range(1, n):
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            displacements.append(math.sqrt(dx * dx + dy * dy))

            ratio = areas[i] / areas[i - 1]
            # Symmetrize: always store >= 1 so both shrinking and growing
            # are treated the same
            area_ratios.append(max(ratio, 1.0 / ratio))

        m = len(displacements)  # m == n - 1
        if m < 2:
            return [records]

        # ── 3. Identify dominant outliers (bimodal gap detection) ─────
        # The key insight: a content change produces a signal that is
        # dramatically larger than any normal variation. We look for
        # values that are a large multiple of the second-highest value,
        # indicating a clear gap in the distribution (bimodality).
        #
        # This avoids the IQR pitfall where even a tight distribution
        # has 1-2 high-ish values that exceed Q3 + k*IQR.

        def dominant_outlier_value(values: List[float]) -> Optional[float]:
            """Return the value if there's a dominant outlier, else None.

            A dominant outlier is a value that is at least `gap_factor` times
            the second-highest value, AND at least `gap_factor` times the
            90th percentile. This ensures it's truly in a class of its own.
            """
            if len(values) < 4:
                return None
            sorted_v = sorted(values)
            max_val = sorted_v[-1]
            second_max = sorted_v[-2]
            p90 = sorted_v[int(len(sorted_v) * 0.90)]

            # The gap factor: outlier must be dramatically larger
            gap_factor = 2.0

            if max_val < 1e-6:
                return None  # all near-zero, no signal

            # Must be > gap_factor times both the second-highest AND p90
            if second_max < 1e-6:
                # All values except max are ~0 — max is clearly an outlier
                # but we need to verify it's not just a single noise spike.
                # Require max > 10 * p90 (which is also ~0 here).
                if max_val > 10.0:
                    return max_val
                return None

            ratio_vs_second = max_val / second_max
            ratio_vs_p90 = max_val / p90 if p90 > 1e-6 else float("inf")

            if ratio_vs_second >= gap_factor and ratio_vs_p90 >= gap_factor:
                return max_val
            return None

        # Find dominant outliers for each signal independently
        dominant_disp = dominant_outlier_value(displacements)
        dominant_area = dominant_outlier_value(area_ratios)

        # ── 4. Find split candidates ─────────────────────────────────
        # Strategy: Area ratio is the primary signal (different person ≈
        # different body size). Displacement is the secondary signal
        # (different person ≈ different position, but startup jitter
        # also causes large displacement).
        #
        # Split candidates require:
        #   - Area ratio must be a dominant outlier (clear size change)
        #   - Displacement must be significant (above median * 3, or above
        #     dominant if available — catches genuine position shifts)
        #
        # This avoids false splits where only displacement spikes (jitter)
        # but area stays consistent.

        if dominant_area is None:
            # No dominant area outlier → no content change detected
            return [records]

        # For displacement, use a permissive threshold: the candidate's
        # displacement must be at least 3x the track's median displacement,
        # ensuring it represents a genuinely large position shift (not just
        # normal motion variance).
        sorted_disps = sorted(displacements)
        median_disp_val = sorted_disps[len(sorted_disps) // 2]
        if median_disp_val < 1e-6:
            disp_min = max(dominant_disp if dominant_disp else 10.0, 10.0)
        else:
            disp_min = max(median_disp_val * 3.0, 10.0)

        # Area must be at least 70% of the dominant outlier value
        area_min = dominant_area * 0.7
        settle_frames = max(3, int(0.05 * n))

        split_indices = []
        for i in range(m):
            # Both signals must exceed their thresholds
            if displacements[i] < disp_min:
                continue
            if area_ratios[i] < area_min:
                continue

            # ── 5. Settle check: centroid and area must stay near new values ─
            if i + settle_frames > m:
                continue  # not enough post-jump frames to verify

            # Pre-jump baseline (average of up to 3 frames before the jump)
            pre_start = max(0, i - 2)
            pre_cx = sum(centroids[j][0] for j in range(pre_start, i + 1)) / (
                i + 1 - pre_start
            )
            pre_cy = sum(centroids[j][1] for j in range(pre_start, i + 1)) / (
                i + 1 - pre_start
            )
            pre_area = sum(areas[j] for j in range(pre_start, i + 1)) / (
                i + 1 - pre_start
            )

            # Post-jump values (average of settle_frames after the jump)
            post_end = min(n, i + 1 + settle_frames)
            post_cx = sum(centroids[j][0] for j in range(i + 1, post_end)) / (
                post_end - i - 1
            )
            post_cy = sum(centroids[j][1] for j in range(i + 1, post_end)) / (
                post_end - i - 1
            )
            post_area = sum(areas[j] for j in range(i + 1, post_end)) / (
                post_end - i - 1
            )

            # Centroid must stay far from pre-jump position
            shift_dist = math.sqrt((post_cx - pre_cx) ** 2 + (post_cy - pre_cy) ** 2)
            if shift_dist < disp_min:
                continue

            # Area must have changed significantly (symmetric ratio)
            area_change = max(post_area / pre_area, pre_area / post_area)
            if area_change < area_min:
                continue

            # Record the split point (index into `records`, not `displacements`)
            split_frame_idx = i + 1
            split_indices.append(
                (split_frame_idx, shift_dist, displacements[i], area_change)
            )

        if not split_indices:
            return [records]

        # ── 6. Apply first valid split and recurse ───────────────────
        for split_at, shift_dist, disp_val, area_chg in split_indices:
            if split_at < min_subtrack_length or n - split_at < min_subtrack_length:
                continue

            part1 = records[:split_at]
            part2 = records[split_at:]

            print(
                f"[Identity] Split track: {n} frames -> "
                f"{len(part1)} + {len(part2)} frames "
                f"(disp_min={disp_min:.1f}px, dom_area={dominant_area:.2f}, "
                f"split_at={split_at}, disp={disp_val:.1f}px, "
                f"area_change={area_chg:.2f}, shift_dist={shift_dist:.1f}px)"
            )

            return IdentityCluster._recursive_split(
                part1, min_subtrack_length
            ) + IdentityCluster._recursive_split(part2, min_subtrack_length)

        # All candidate splits would produce too-short subtracks
        return [records]

    @staticmethod
    def _median_bbox(bboxes):
        """Compute median bounding box from a list."""
        if not bboxes:
            return None
        xs = [b.x1 for b in bboxes] + [b.x2 for b in bboxes]
        ys = [b.y1 for b in bboxes] + [b.y2 for b in bboxes]
        mid_x = sorted(xs)[len(xs) // 2]
        mid_y = sorted(ys)[len(ys) // 2]
        return BoundingBox(mid_x - 1, mid_y - 1, mid_x + 1, mid_y + 1)

    # Spatial distance threshold (pixels) for temporal blocker validation.
    # A track in the gap is only considered a real blocker if its median
    # centroid is within this distance of the candidate track. This
    # tolerates natural centroid jitter of the same person across frames
    # while filtering out spatially distinct people.
    SPATIAL_BLOCKER_THRESHOLD = 200.0

    @staticmethod
    def _track_median_centroid(records):
        """Compute median centroid (cx, cy) from FrameRecords.

        Args:
            records: List of FrameRecord objects with x1, y1, x2, y2.

        Returns:
            (cx, cy) tuple or None if records is empty.
        """
        if not records:
            return None
        cxs = sorted((r.x2 + r.x1) / 2.0 for r in records)
        cys = sorted((r.y2 + r.y1) / 2.0 for r in records)
        return (cxs[len(cxs) // 2], cys[len(cys) // 2])

    def _has_temporal_blocker(
        self,
        tid: int,
        actor_member_tids: List[int],
        track_time_ranges: Dict[int, Tuple[int, int]],
        valid_track_ids: set,
        track_records: Dict[int, list],
    ) -> bool:
        """Check if any other track exists between tid and actor members.

        A track B is considered a blocker only when ALL of the following hold:
          1. B's detection range falls entirely within the temporal gap
             between tid and the candidate actor.
          2. B's median centroid is spatially close (within
             SPATIAL_BLOCKER_THRESHOLD) to the candidate track (tid) —
             indicating B is the "same spatial region" as the candidate.

        The blocker only needs to be near the candidate track, not near the
        actor. This correctly handles the case where the candidate and actor
        are at different positions (e.g. different people), since the
        blocker's spatial proximity to the candidate proves someone else was
        present in the candidate's region during the gap.

        Args:
            tid: Track ID being evaluated.
            actor_member_tids: Track IDs belonging to the candidate actor.
            track_time_ranges: Mapping from track ID to (first_frame, last_frame).
            valid_track_ids: Set of all valid track IDs.
            track_records: Mapping from track ID to FrameRecord list.

        Returns:
            True if at least one spatially-proximate blocker track is found
            in the gap.
        """
        t_lo, t_hi = track_time_ranges[tid]

        # Determine the actor's time boundaries
        a_lo = min(track_time_ranges[mt][0] for mt in actor_member_tids)
        a_hi = max(track_time_ranges[mt][1] for mt in actor_member_tids)

        # Determine the gap boundaries
        if t_lo > a_hi:
            gap_start, gap_end = a_hi, t_lo
        elif a_lo > t_hi:
            gap_start, gap_end = t_hi, a_lo
        else:
            return False  # overlapping, not applicable

        # Pre-compute median centroid for the candidate track
        tid_centroid = self._track_median_centroid(track_records.get(tid, []))

        threshold = self.SPATIAL_BLOCKER_THRESHOLD

        # Check if any other track falls entirely within this gap
        # AND is spatially close to the candidate track (tid)
        for other_tid in valid_track_ids:
            if other_tid == tid or other_tid in actor_member_tids:
                continue
            o_lo, o_hi = track_time_ranges[other_tid]
            if o_lo < gap_start or o_hi > gap_end:
                continue  # not fully within the gap

            # Spatial check: blocker must be near the candidate track (tid)
            blocker_centroid = self._track_median_centroid(
                track_records.get(other_tid, [])
            )
            if blocker_centroid is None or tid_centroid is None:
                continue  # can't determine spatial relation

            dist_to_tid = (
                (blocker_centroid[0] - tid_centroid[0]) ** 2
                + (blocker_centroid[1] - tid_centroid[1]) ** 2
            ) ** 0.5

            if dist_to_tid > threshold:
                print(
                    f"[Identity]     Blocker {other_tid} excluded: "
                    f"dist_to_tid={dist_to_tid:.0f} "
                    f"(threshold={threshold:.0f})"
                )
                continue

            return True

        return False

    @staticmethod
    def _compute_temporal_confidence(
        gap: int,
        max_lost_frames: int,
        has_blocker: bool,
    ) -> float:
        """Compute temporal confidence for merging two non-adjacent tracks.

        Based on the principle: if no other person appears between two
        non-overlapping tracks, it is more likely the same person left
        and returned.

        Uses a Gaussian-like decay centered at max_lost_frames — confidence
        is highest for moderate gaps (1–2× max_lost_frames) and decays
        for very large gaps.

        Args:
            gap: Temporal gap in frames between the two tracks.
            max_lost_frames: Tracker's lost-frame tolerance threshold.
            had_blocker: Whether a blocker track exists in the gap.

        Returns:
            Confidence score between 0.0 and 1.0. Higher means more likely
            the same person.
        """
        if has_blocker:
            return 0.0  # other people in between, likely different person

        if gap <= max_lost_frames:
            return 0.0  # tracker had chance to merge, no temporal signal

        # Gaussian-like decay centered at max_lost_frames
        # No-blocker boost: when no other person appears in the gap,
        # increase confidence by 1.5× (capped at 1.0).
        sigma = max_lost_frames * 3
        conf = math.exp(-(((gap - max_lost_frames) / sigma) ** 2))
        conf *= 1.5

        return min(conf, 1.0)

    @staticmethod
    def _bbox_iou(a, b):
        """Compute IoU between two BoundingBoxes."""
        if a is None or b is None:
            return 0.0
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
        area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / (union + 1e-8)
