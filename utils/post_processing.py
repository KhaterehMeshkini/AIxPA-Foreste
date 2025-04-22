import numpy as np
from scipy.ndimage import label, binary_dilation
import utils.filemanager as fm
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm import tqdm


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

def _process_hole(label_id, labeled_holes, filled_change_array, updated_probability_array, max_hole_size, no_change_value):
    hole_mask = (labeled_holes == label_id)
    hole_size = np.count_nonzero(hole_mask)

    if hole_size > max_hole_size:
        return None  # Skip this hole

    dilated = binary_dilation(hole_mask)
    boundary_mask = dilated & (~hole_mask)

    boundary_values = filled_change_array[boundary_mask]
    boundary_probs = updated_probability_array[boundary_mask]

    valid_mask = boundary_values != no_change_value

    if np.any(valid_mask):
        mean_change = int(np.mean(boundary_values[valid_mask]))  # Using int(mean) for change
        mean_prob = np.mean(boundary_probs[valid_mask])
    else:
        mean_change = no_change_value
        mean_prob = 0.0

    return label_id, mean_change, mean_prob

def fill_small_holes_and_update_probabilities(change_array, probability_array, max_hole_size=16, no_change_value=0, max_workers=8):
    change_array = change_array.astype(np.uint8)
    probability_array = probability_array.astype(np.float16)

    filled_change_array = change_array.copy()
    updated_probability_array = probability_array.copy()

    nodata_mask = (filled_change_array == no_change_value)
    labeled_holes, num_holes = label(nodata_mask)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for label_id in range(1, num_holes + 1):
            futures.append(executor.submit(
                _process_hole,
                label_id, labeled_holes,
                filled_change_array,
                updated_probability_array,
                max_hole_size,
                no_change_value
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Filling holes"):
            result = future.result()
            if result is not None:
                label_id, mean_change, mean_prob = result
                hole_mask = (labeled_holes == label_id)
                filled_change_array[hole_mask] = mean_change
                updated_probability_array[hole_mask] = mean_prob

    return filled_change_array.astype(np.uint8), updated_probability_array.astype(np.float16)
    

    
    



    



