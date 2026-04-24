"""Face recognition and identity clustering module.

Uses InsightFace for face embedding extraction and cosine similarity
clustering to merge tracks belonging to the same actor.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from pipeline.tracker import FrameRecord
from pipeline.detector import BoundingBox
from core.config import DEFAULT_FACE_THRESHOLD, DEFAULT_MIN_FACE_CONFIDENCE


class IdentityCluster:
    """Identity clustering using InsightFace face embeddings."""

    def __init__(self, threshold: float = DEFAULT_FACE_THRESHOLD):
        self.threshold = threshold
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load InsightFace model."""
        if self._loaded:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis
            import signal

            # Set proxy for model download
            proxy = os.environ.get("http_proxy", os.environ.get("HTTP_PROXY", ""))
            if proxy:
                os.environ.setdefault("HTTP_PROXY", proxy)
                os.environ.setdefault("HTTPS_PROXY", proxy)

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
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from a cropped region of a frame.
        Tries multiple crop sizes to maximize face detection chances.

        Args:
            frame: Full BGR frame
            bbox: Bounding box of the person

        Returns:
            Face embedding vector or None if no face detected
        """
        self._ensure_loaded()
        if self.model is None:
            return None

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

            return best_face.embedding

        # Fallback: try the full frame (for cases where bbox is very inaccurate)
        faces = self.model.get(frame)
        if faces:
            best_face = max(faces, key=lambda f: f.det_score)
            if best_face.det_score >= DEFAULT_MIN_FACE_CONFIDENCE:
                return best_face.embedding

        return None

    def _get_track_embedding(self, track_records, frames):
        """Get aggregated face embedding for a track by trying multiple frames.
        
        Improvements over previous version:
        - Samples up to 30 frames instead of 10
        - Prioritizes frames where mask area is largest (person closest to camera)
        - For long tracks, samples uniformly across entire duration
        """
        embeddings = []
        n = len(track_records)
        if n == 0:
            return None

        # Sort records by mask area (descending) to prioritize frames where person is largest
        sorted_records = sorted(
            track_records,
            key=lambda r: (r.x2 - r.x1) * (r.y2 - r.y1),
            reverse=True,
        )

        # Sample up to 30 frames, prioritizing largest area frames
        max_samples = 30
        if n <= max_samples:
            # Try all frames
            sample_records = sorted_records
        else:
            # Take top frames by area, but also ensure temporal coverage
            # Take top 20 by area + 10 evenly spaced from full track
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

        for rec in sample_records:
            frame = frames.get(rec.frame_idx, None)
            if frame is None:
                continue
            bbox = BoundingBox(rec.x1, rec.y1, rec.x2, rec.y2)
            emb = self._get_face_embedding(frame, bbox)
            if emb is not None:
                embeddings.append((emb, rec))

        if not embeddings:
            return None

        # Use weighted average: higher weight for larger area frames (closer to camera = better face quality)
        if len(embeddings) == 1:
            avg_emb = embeddings[0][0]
        else:
            # Simple average of all successful embeddings
            # (InsightFace embeddings are already normalized, so average + re-normalize works well)
            avg_emb = np.mean([e[0] for e in embeddings], axis=0)
        
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
        return avg_emb

    def cluster_tracks(
        self,
        track_records: Dict[int, List[FrameRecord]],
        frames: Dict[int, np.ndarray],
        min_track_length: int = 5,
    ) -> Dict[int, str]:
        """
        Cluster tracks into actor identities.
        
        Uses connected components clustering on face similarity graph,
        with temporal constraints to prevent merging tracks that are
        temporally adjacent (same person appearing in different parts of video).

        Args:
            track_records: {track_id: [FrameRecord, ...]}
            frames: {frame_idx: numpy_frame} - all frames needed for face extraction
            min_track_length: Minimum number of frames for a track to be considered

        Returns:
            {track_id: "actor_0", ...}
        """
        self._ensure_loaded()

        # Filter out short tracks
        valid_tracks = {
            tid: recs
            for tid, recs in track_records.items()
            if len(recs) >= min_track_length
        }

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
            emb = self._get_track_embedding_with_count(recs, frames)
            track_embeddings[tid] = emb[0]
            face_detection_counts[tid] = emb[1]
        
        # Debug: print face detection stats
        tracks_with_faces = sum(1 for e in track_embeddings.values() if e is not None)
        print(f"[Identity] Face detection: {tracks_with_faces}/{len(valid_tracks)} tracks have face embeddings")
        for tid, count in face_detection_counts.items():
            if count > 0:
                print(f"[Identity]   Track {tid}: {count} faces detected")

        # Build similarity matrix
        tracks_with_embs = [tid for tid, emb in track_embeddings.items() if emb is not None]
        n_tracks = len(tracks_with_embs)
        
        print(f"[Identity] Building similarity matrix ({n_tracks}x{n_tracks})...")
        sim_matrix = np.zeros((n_tracks, n_tracks))
        for i, tid_i in enumerate(tracks_with_embs):
            for j, tid_j in enumerate(tracks_with_embs):
                if i >= j:
                    continue
                sim = float(np.dot(track_embeddings[tid_i], track_embeddings[tid_j]))
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # Use connected components clustering with temporal constraints
        # Two tracks are connected if:
        # 1. Face similarity >= threshold
        # 2. They are temporally adjacent (gap <= max_gap) OR similarity is very high
        
        # Compute temporal adjacency
        track_time_ranges: Dict[int, Tuple[int, int]] = {}
        for tid, recs in valid_tracks.items():
            frame_indices = [r.frame_idx for r in recs]
            track_time_ranges[tid] = (min(frame_indices), max(frame_indices))

        max_gap = 100  # Maximum frame gap for temporal adjacency
        high_sim_threshold = 0.75  # Higher threshold for non-adjacent tracks

        # Build adjacency list
        adj: Dict[int, set] = {tid: set() for tid in tracks_with_embs}
        
        merge_log: List[Tuple[int, int, float]] = []
        
        for i, tid_i in enumerate(tracks_with_embs):
            for j in range(i + 1, n_tracks):
                tid_j = tracks_with_embs[j]
                sim = sim_matrix[i, j]
                
                if sim < self.threshold:
                    continue
                
                # Check temporal adjacency
                lo_i, hi_i = track_time_ranges[tid_i]
                lo_j, hi_j = track_time_ranges[tid_j]
                
                # Compute gap (0 if overlapping)
                gap = max(0, max(lo_i, lo_j) - min(hi_i, hi_j))
                
                # Determine if should merge
                if gap <= max_gap:
                    # Temporally adjacent: use normal threshold
                    should_merge = True
                elif sim >= high_sim_threshold:
                    # Very high similarity: merge even if not adjacent
                    # (same person appearing in different parts of video)
                    should_merge = True
                else:
                    # Not adjacent and not high enough similarity
                    should_merge = False
                
                if should_merge:
                    adj[tid_i].add(tid_j)
                    adj[tid_j].add(tid_i)
                    merge_log.append((tid_i, tid_j, sim))

        # Find connected components using Union-Find
        parent: Dict[int, int] = {tid: tid for tid in tracks_with_embs}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path compression
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for tid_i in tracks_with_embs:
            for tid_j in adj[tid_i]:
                union(tid_i, tid_j)
        
        # Group by component root
        components: Dict[int, List[int]] = defaultdict(list)
        for tid in tracks_with_embs:
            components[find(tid)].append(tid)
        
        # Assign actor IDs
        track_to_actor: Dict[int, str] = {}
        actor_id_counter = 0
        
        # Sort components by first track ID for deterministic ordering
        sorted_components = sorted(components.values(), key=lambda comps: min(comps))
        
        for comp in sorted_components:
            actor_id = f"actor_{actor_id_counter}"
            actor_id_counter += 1
            for tid in comp:
                track_to_actor[tid] = actor_id
            
            # Log merge info
            if len(comp) > 1:
                print(f"[Identity] Merged {len(comp)} tracks into {actor_id}: {comp}")
                for i, tid_i in enumerate(comp):
                    for tid_j in comp[i+1:]:
                        lo_i, hi_i = track_time_ranges[tid_i]
                        lo_j, hi_j = track_time_ranges[tid_j]
                        gap = max(0, max(lo_i, lo_j) - min(hi_i, hi_j))
                        # Find similarity
                        idx_i = tracks_with_embs.index(tid_i)
                        idx_j = tracks_with_embs.index(tid_j)
                        sim = sim_matrix[idx_i, idx_j]
                        print(f"[Identity]   {tid_i}({lo_i}-{hi_i}) <-> {tid_j}({lo_j}-{hi_j}): sim={sim:.3f}, gap={gap}")

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
            actor_id_counter = max(
                int(a.split("_")[1]) for a in all_actors if a.startswith("actor_")
            ) + 1 if all_actors else 0
        
        # Renumber actor IDs sequentially
        unique_actors = sorted(set(track_to_actor.values()))
        actor_map = {aid: idx for idx, aid in enumerate(unique_actors)}
        track_to_actor = {tid: f"actor_{actor_map[old_name]}" for tid, old_name in track_to_actor.items()}

        print(f"[Identity] Final: {len(set(track_to_actor.values()))} unique actors")
        return track_to_actor

    def _get_track_embedding_with_count(self, track_records, frames):
        """Get aggregated face embedding for a track, returning (embedding, face_count)."""
        embeddings = []
        n = len(track_records)
        if n == 0:
            return None, 0

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
        for rec in sample_records:
            frame = frames.get(rec.frame_idx, None)
            if frame is None:
                continue
            bbox = BoundingBox(rec.x1, rec.y1, rec.x2, rec.y2)
            emb = self._get_face_embedding(frame, bbox)
            if emb is not None:
                embeddings.append(emb)
                face_count += 1

        if not embeddings:
            return None, face_count

        # Simple average of all successful embeddings
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
        return avg_emb, face_count

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
    def _median_bbox(bboxes):
        """Compute median bounding box from a list."""
        if not bboxes:
            return None
        xs = [b.x1 for b in bboxes] + [b.x2 for b in bboxes]
        ys = [b.y1 for b in bboxes] + [b.y2 for b in bboxes]
        mid_x = sorted(xs)[len(xs) // 2]
        mid_y = sorted(ys)[len(ys) // 2]
        return BoundingBox(mid_x - 1, mid_y - 1, mid_x + 1, mid_y + 1)

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
