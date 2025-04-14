import numpy as np
from scipy.ndimage import label, binary_fill_holes
from scipy.ndimage.filters import uniform_filter
import filemanager as fm
from scipy.ndimage import convolve

#--------------------------------------------------------#
# postprocessing

def remove_isolated_pixels(change_array, probability_array, area_threshold=16):
    """
    Remove isolated pixels from the change map and update probabilities.
    Non-zero values in the change array represent "changed" pixels, while zero represents "no change".
    
    Parameters:
    - change_array: A 2D numpy array with values representing change (non-zero for changes).
    - probability_array: A 2D numpy array with probability values for changes.
    - area_threshold: The minimum area in pixels to retain a connected region.
    
    Returns:
    - updated_change_array: Change array with isolated pixels removed.
    - updated_probability_array: Probability array with corresponding isolated pixel probabilities removed.
    """
    # Create copies of the arrays to manipulate
    updated_change_array = change_array.copy()
    updated_probability_array = probability_array.copy()

    # Create a mask for the regions considered as change (non-zero values are change)
    change_mask = updated_change_array != 0

    # Label connected components in the change map
    labeled_array, num_features = label(change_mask)

    # Remove isolated regions below the area threshold
    for label_num in range(1, num_features + 1):
        component_mask = (labeled_array == label_num)
        if np.sum(component_mask) < area_threshold:
            updated_change_array[component_mask] = 0  # Use NaN to indicate removed changes
            updated_probability_array[component_mask] = 0  # Remove probability for those pixels

    return updated_change_array, updated_probability_array


def fill_small_holes_and_update_probabilities(change_array, probability_array, max_hole_size=16, no_change_value=0):
    """
    Fill small holes (nodata values) in the change map and assign probabilities to the filled pixels.
    Holes larger than `max_hole_size` are ignored. 

    Parameters:
    - change_array: A 2D numpy array with values representing change (non-zero for changes, zero for no change).
    - probability_array: A 2D numpy array with probability values (0–1) for changes.
    - max_hole_size: Maximum size of holes (in pixels) to fill.
    - no_change_value: Value to represent 'no change' (default is 0).
    
    Returns:
    - filled_change_array: Change array with small nodata values filled based on neighboring values.
    - updated_probability_array: Probability array with assigned values for filled pixels.
    """
    # Copy arrays to avoid modifying the originals
    filled_change_array = change_array.copy()
    updated_probability_array = probability_array.copy()

    # Identify nodata regions (NaN values or a specified value for no data)
    nodata_mask = np.isnan(filled_change_array) | (filled_change_array == no_change_value)

    # Label connected components of nodata regions
    labeled_holes, num_holes = label(nodata_mask)

    # Process each hole
    for hole_label in range(1, num_holes + 1):
        # Create a mask for the current hole
        hole_mask = (labeled_holes == hole_label)
        hole_size = np.sum(hole_mask)

        # Skip holes larger than the threshold
        if hole_size > max_hole_size:
            continue

        # Get indices of the current hole
        rows, cols = np.where(hole_mask)

        # Process the hole pixels
        for row, col in zip(rows, cols):
            # Extract a 9x9 neighborhood around the pixel
            row_min, row_max = max(0, row - 4), min(change_array.shape[0], row + 5)
            col_min, col_max = max(0, col - 4), min(change_array.shape[1], col + 5)

            neighborhood_change = filled_change_array[row_min:row_max, col_min:col_max]
            neighborhood_prob = updated_probability_array[row_min:row_max, col_min:col_max]

            # Get valid neighbors (non-zero values) in the change array
            valid_mask = neighborhood_change != 0
            valid_changes = neighborhood_change[valid_mask]

            # Fill nodata pixel in the change array
            if valid_changes.size > 0:
                # Assign based on the majority of neighbors (or other strategies if needed)
                filled_change_array[row, col] = np.mean(valid_changes)  # For example, use the average

                # Assign probability as the average of valid neighboring probabilities
                valid_probabilities = neighborhood_prob[valid_mask]
                updated_probability_array[row, col] = np.mean(valid_probabilities)
            else:
                # If no valid neighbors exist, default to filling the pixel with `no_change_value`
                filled_change_array[row, col] = no_change_value
                updated_probability_array[row, col] = 0  # Default probability for filled pixels

    return filled_change_array, updated_probability_array