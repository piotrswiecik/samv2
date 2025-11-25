import numpy as np
import cv2


def clean_mask(binary_mask):
    """
    Turns a 'sandy' pixel cloud into a solid, contiguous vessel.
    """
    # 1. Morphological Closing (Fills micro-gaps)
    # Kernel size determines how big a gap to fill. 5x5 is usually safe for 1024px.
    kernel = np.ones((5, 5), np.uint8)
    solid_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. Keep Largest Component (Removes floating dust)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(solid_mask, connectivity=8)
    
    if num_labels <= 1:
        return solid_mask # No fg objects found
        
    # stats column 4 is Area. Index 0 is background, so we start from 1.
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    cleaned_mask = np.zeros_like(solid_mask)
    cleaned_mask[labels == largest_label] = 1
    
    return cleaned_mask