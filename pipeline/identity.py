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
            proxy = os.environ.get('http_proxy', os.environ.get('HTTP_PROXY', ''))
            if proxy:
                os.environ.setdefault('HTTP_PROXY', proxy)
                os.environ.setdefault('HTTPS_PROXY', proxy)
            
            model_dir = os.path.expanduser("~/.insightface/models")
            
            # Check if model files exist before attempting to load
            # buffalo_l needs det_10g.onnx and w600k_r50.onnx
            required_files = ['1k3d68.onnx', '2d106det.onnx', 'det_10g.onnx', 'genderage.onnx', 'w600k_mbf.onnx', 'w600k_r50.onnx']
            model_path = os.path.join(model_dir, "buffalo_l")
            has_all = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
            
            if not has_all:
                print("[Identity] InsightFace buffalo_l model not found locally, skipping face clustering")
                print("[Identity] Run: python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)\"")
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
    
    def _get_face_embedding(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """
        Extract face embedding from a cropped region of a frame.
        
        Args:
            frame: Full BGR frame
            bbox: Bounding box of the person
            
        Returns:
            Face embedding vector or None if no face detected
        """
        self._ensure_loaded()
        if self.model is None:
            return None
        
        # Crop person region with some padding
        h, w = frame.shape[:2]
        pad_x = int(bbox.width * 0.2)
        pad_y = int(bbox.height * 0.3)
        x1 = max(0, int(bbox.x1) - pad_x)
        y1 = max(0, int(bbox.y1) - pad_y)
        x2 = min(w, int(bbox.x2) + pad_x)
        y2 = min(h, int(bbox.y2) + pad_y)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        faces = self.model.get(crop)
        if not faces:
            return None
        
        # Return the embedding of the face with highest detection confidence
        best_face = max(faces, key=lambda f: f.det_score)
        if best_face.det_score < DEFAULT_MIN_FACE_CONFIDENCE:
            return None
        
        return best_face.embedding
    
    def _get_track_embedding(self, track_records: List[FrameRecord], frames: np.ndarray) -> Optional[np.ndarray]:
        """
        Get aggregated face embedding for a track by sampling key frames.
        
        Args:
            track_records: List of FrameRecord for this track
            frames: All sampled frames (indexed by original frame index)
            
        Returns:
            Average face embedding or None
        """
        embeddings = []
        
        # Sample up to 5 key frames from the track
        n = len(track_records)
        if n == 0:
            return None
        
        if n <= 5:
            indices = list(range(n))
        else:
            step = n // 5
            indices = list(range(0, n, step))[:5]
        
        for idx in indices:
            rec = track_records[idx]
            # Map frame_idx to index in frames array
            # frames is indexed by the sampling, so we need the original frame mapping
            # We'll store the frame array alongside the records
            # For now, assume frames are accessible by frame_idx
            # This will be handled by the caller
            frame = frames.get(rec.frame_idx, None)
            if frame is None:
                continue
            
            bbox = BoundingBox(rec.x1, rec.y1, rec.x2, rec.y2)
            emb = self._get_face_embedding(frame, bbox)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        # Average embeddings and normalize
        avg_emb = np.mean(embeddings, axis=0)
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
            tid: recs for tid, recs in track_records.items()
            if len(recs) >= min_track_length
        }
        
        if not valid_tracks:
            return {}
        
        # If no face model, assign each track a unique actor ID
        if self.model is None:
            return {tid: f"actor_{i}" for i, tid in enumerate(valid_tracks.keys())}
        
        # Extract face embeddings for each track
        track_embeddings: Dict[int, Optional[np.ndarray]] = {}
        for tid, recs in valid_tracks.items():
            emb = self._get_track_embedding(recs, frames)
            track_embeddings[tid] = emb
        
        # Greedy clustering by cosine similarity
        actor_id_counter = 0
        track_to_actor: Dict[int, str] = {}
        # actor_id -> list of embeddings belonging to this actor
        actor_embeddings: Dict[str, List[np.ndarray]] = {}
        
        # Sort tracks by embedding quality (non-None first)
        sorted_tracks = sorted(
            valid_tracks.keys(),
            key=lambda tid: 0 if track_embeddings[tid] is not None else 1,
        )
        
        for tid in sorted_tracks:
            emb = track_embeddings[tid]
            if emb is None:
                # No face detected, give unique ID
                actor_id = f"actor_{actor_id_counter}"
                actor_id_counter += 1
                track_to_actor[tid] = actor_id
                continue
            
            # Find best matching existing actor
            best_similarity = -1
            best_actor = None
            
            for actor_id, embs in actor_embeddings.items():
                # Average similarity to all embeddings in this actor group
                sims = [float(np.dot(emb, e)) for e in embs]
                avg_sim = np.mean(sims)
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_actor = actor_id
            
            if best_similarity >= self.threshold:
                # Merge into existing actor
                track_to_actor[tid] = best_actor
                actor_embeddings[best_actor].append(emb)
            else:
                # Create new actor
                actor_id = f"actor_{actor_id_counter}"
                actor_id_counter += 1
                track_to_actor[tid] = actor_id
                actor_embeddings[actor_id] = [emb]
        
        return track_to_actor
