import numpy as np
from app.logger import logger

# These thresholds are just examples and will need to be tuned based on real data.
MAX_SPEED = 15  # m/s, very fast for a deer
MIN_LENGTH = 10  # meters
MAX_TURN_ANGLE = 90 # degrees

def _get_turn_angle(trj):
    """Calculate turn angle."""
    dx = np.diff(trj.x)
    dy = np.diff(trj.y)
    angles = np.arctan2(dy, dx)
    turn_angles = np.diff(angles)
    return np.rad2deg(turn_angles)

def is_biologically_plausible(length, speed, trj) -> bool:
    """
    Validates a trackway based on biological movement patterns.
    """
    if length < MIN_LENGTH:
        logger.debug(f"Trackway rejected: length {length} < {MIN_LENGTH}")
        return False

    if not speed.empty and speed.max() > MAX_SPEED:
        logger.debug(f"Trackway rejected: max speed {speed.max()} > {MAX_SPEED}")
        return False

    # Calculate turning angles
    if len(trj) < 3:
        return True # Not enough points to calculate a turning angle

    angles = _get_turn_angle(trj)
    if angles.size > 0 and np.abs(angles).max() > MAX_TURN_ANGLE:
        logger.debug(f"Trackway rejected: max turn angle {np.abs(angles).max()} > {MAX_TURN_ANGLE}")
        return False

    return True
