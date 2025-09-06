import numpy as np
from app.logger import logger

# These thresholds are just examples and will need to be tuned based on real data.
MAX_SPEED = 15  # m/s, very fast for a deer
MIN_LENGTH = 10  # meters
MAX_TURN_ANGLE = 90  # degrees
MAX_TORTUOSITY = 5  # Dimensionless
MIN_COMMUTING_SPEED = 5 # m/s
MAX_COMMUTING_TORTUOSITY = 1.2 # Dimensionless

def _calculate_end_to_end_displacement(trj):
    """Calculate the straight-line distance between the start and end of a trajectory."""
    if len(trj) < 2:
        return 0
    start_point = trj.iloc[0]
    end_point = trj.iloc[-1]
    return np.sqrt((end_point.x - start_point.x)**2 + (end_point.y - start_point.y)**2)

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

    # Calculate tortuosity
    displacement = _calculate_end_to_end_displacement(trj)
    if displacement > 0:
        tortuosity = length / displacement
        if tortuosity > MAX_TORTUOSITY:
            logger.debug(f"Trackway rejected: tortuosity {tortuosity} > {MAX_TORTUOSITY}")
            return False

    # TODO: Add more advanced validation based on ecological literature.
    # For example, check for patterns indicative of specific behaviors (e.g., foraging, commuting).
    # This could involve analyzing step length distributions, turning angle correlations, etc.

    # Example of behavior-specific validation: Commuting
    if displacement > 0:
        tortuosity = length / displacement
        if not speed.empty and speed.mean() > MIN_COMMUTING_SPEED:
            if tortuosity < MAX_COMMUTING_TORTUOSITY:
                logger.debug("Trackway identified as plausible commuting behavior.")

    return True
