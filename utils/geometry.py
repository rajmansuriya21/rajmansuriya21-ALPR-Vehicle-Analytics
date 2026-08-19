"""
Geometry utilities for line crossing detection.

Provides mathematical functions for:
- Line segment intersection testing
- Crossing direction determination (via cross product)
- Point-to-line side classification
"""

from typing import Tuple

Point = Tuple[float, float]


def cross_product_2d(o: Point, a: Point, b: Point) -> float:
    """
    Compute the cross product of vectors OA and OB.

    Returns:
        Positive if counter-clockwise, negative if clockwise, zero if collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Check if line segment p1–p2 intersects with line segment p3–p4.

    Uses the cross-product orientation method for robust detection.

    Args:
        p1, p2: Endpoints of the first segment (e.g., vehicle movement vector).
        p3, p4: Endpoints of the second segment (e.g., virtual line).

    Returns:
        True if the segments intersect (proper or touching).
    """
    d1 = cross_product_2d(p3, p4, p1)
    d2 = cross_product_2d(p3, p4, p2)
    d3 = cross_product_2d(p1, p2, p3)
    d4 = cross_product_2d(p1, p2, p4)

    # Standard case: segments straddle each other
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear / touching cases
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True

    return False


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """Check if point r lies on segment p–q (assumes collinearity)."""
    return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
            min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def crossing_direction(
    prev_pos: Point,
    curr_pos: Point,
    line_start: Point,
    line_end: Point,
) -> str:
    """
    Determine the direction of a line crossing.

    Uses the cross product of the line vector and the movement vector
    to determine which direction the object crossed the line.

    Args:
        prev_pos: Previous position of the object.
        curr_pos: Current position of the object.
        line_start: Start point of the virtual line.
        line_end: End point of the virtual line.

    Returns:
        "left_to_right" or "right_to_left" relative to the line direction.
    """
    # Line direction vector
    line_dx = line_end[0] - line_start[0]
    line_dy = line_end[1] - line_start[1]

    # Movement vector
    move_dx = curr_pos[0] - prev_pos[0]
    move_dy = curr_pos[1] - prev_pos[1]

    # Cross product determines which side the movement came from
    cross = line_dx * move_dy - line_dy * move_dx

    if cross > 0:
        return "left_to_right"
    else:
        return "right_to_left"


def point_side_of_line(point: Point, line_start: Point, line_end: Point) -> int:
    """
    Determine which side of a line a point is on.

    Args:
        point: The point to check.
        line_start: Start of the line.
        line_end: End of the line.

    Returns:
        +1 if on the left side, -1 if on the right side, 0 if on the line.
    """
    cross = cross_product_2d(line_start, line_end, point)
    if cross > 0:
        return 1
    elif cross < 0:
        return -1
    return 0


def centroid_from_bbox(bbox: list) -> Point:
    """Calculate centroid (center point) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
