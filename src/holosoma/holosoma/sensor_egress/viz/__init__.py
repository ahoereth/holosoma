"""Local-visualization sensor egress (cv2 window / mp4).

Imports cv2 + video utils at module top — loaded only when a ``VizEgressConfig.egress_cls``
fires (i.e. when the viz egress is selected), so the egress package stays cv2-free otherwise.
"""
