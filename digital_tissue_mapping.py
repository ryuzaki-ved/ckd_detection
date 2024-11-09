import cv2
import numpy as np
import os

def analyze_tissue_image(image_path, baseline_area=None):
    # Load the tissue image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Preprocess: thresholding to separate tissue area (assuming darker tissues)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to determine tissue size
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze each tissue in the image
    tissue_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        tissue_areas.append(area)
    
    # Compare to baseline area if provided
    results = []
    if baseline_area is not None:
        for area in tissue_areas:
            difference = (area - baseline_area) / baseline_area * 100  # Percent difference
            if area > baseline_area:
                status = "Swollen"
            elif area < baseline_area:
                status = "Shrunk"
            else:
                status = "Unchanged"
            results.append((status, difference))
    else:
        results = [("Baseline measurement", area) for area in tissue_areas]
    
    return results

def process_image_folder(folder_path, baseline_area=None):
    tissue_status = {"Swollen": 0, "Shrunk": 0, "Unchanged": 0}
    all_differences = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            results = analyze_tissue_image(image_path, baseline_area)
            
            if results:
                for status, difference in results:
                    if status in tissue_status:
                        tissue_status[status] += 1
                    if isinstance(difference, (int, float)):
                        all_differences.append(difference)
    
    # Summary
    print("Tissue Status Summary:")
    for status, count in tissue_status.items():
        print(f"{status}: {count}")
    if all_differences:
        avg_difference = sum(all_differences) / len(all_differences)
        print(f"Average Size Difference: {avg_difference:.2f}%")
    else:
        print("No differences calculated.")
  
# Example usage
baseline_area = 5000  # Set a baseline area for comparison
folder_path = "/path/to/tissue/images"  # Folder containing tissue images
process_image_folder(folder_path, baseline_area)
