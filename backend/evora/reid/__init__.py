"""Cross-camera identity (M2): appearance features, topology, association, paths and query by example."""
from evora.reid.associate import link_global_ids
from evora.reid.paths import path_for
from evora.reid.similar import similar_tracks

__all__ = ["link_global_ids", "path_for", "similar_tracks"]
