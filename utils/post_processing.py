import numpy as np
from scipy.ndimage import label, binary_dilation
import utils.filemanager as fm
from scipy.ndimage import label



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
    # Ensure efficient dtypes
    change_array = change_array.astype(np.uint8)
    probability_array = probability_array.astype(np.float16)

    # Binary mask and label connected components
    change_mask = change_array != 0
    labeled_array, num_features = label(change_mask, structure=np.ones((3, 3)))  # 8-connectivity

    # Count pixel area per component using bincount
    label_sizes = np.bincount(labeled_array.ravel())

    # Skip label 0 (background)
    small_labels = np.where(label_sizes < area_threshold)[0]
    small_labels = small_labels[small_labels != 0]

    # Mask small regions in a single pass
    if small_labels.size > 0:
        small_mask = np.isin(labeled_array, small_labels)
        change_array[small_mask] = 0
        probability_array[small_mask] = 0

    return change_array, probability_array



    
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

    change_array = change_array.astype(np.uint8)
    probability_array = probability_array.astype(np.float16)

    filled_change_array = change_array.copy()
    updated_probability_array = probability_array.copy()

    nodata_mask = (filled_change_array == no_change_value)
    labeled_holes, num_holes = label(nodata_mask)

    for i in range(1, num_holes + 1):
        hole_mask = labeled_holes == i
        hole_size = np.sum(hole_mask)

        if hole_size > max_hole_size:
            continue

        # Dilate hole to find boundary
        dilated_mask = binary_dilation(hole_mask)
        boundary_mask = dilated_mask & (~hole_mask)

        # Get valid neighbors on the boundary
        boundary_changes = filled_change_array[boundary_mask]
        boundary_probs = updated_probability_array[boundary_mask]
        valid_neighbors = boundary_changes != no_change_value

        if np.any(valid_neighbors):
            mean_change = int(np.mean(boundary_changes[valid_neighbors]))
            mean_prob = np.mean(boundary_probs[valid_neighbors])
        else:
            mean_change = no_change_value
            mean_prob = 0.0

        filled_change_array[hole_mask] = mean_change
        updated_probability_array[hole_mask] = mean_prob

    return filled_change_array.astype(np.uint8), updated_probability_array.astype(np.float16)


    
    



    



