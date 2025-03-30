import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math

def refine_boundaries(image_path, output_path):
    """
    Basic contour-based boundary refinement with adaptive thresholding
    """
   
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
   
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
   
    min_area = 50  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    
    mask = np.zeros_like(gray)
    
   
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    
    
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    
    
    refined_contours, _ = cv2.findContours(
        refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    
    smoothed_contours = []
    for contour in refined_contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(approx)
    
   
    result = img_rgb.copy()
    
    
    cv2.drawContours(result, smoothed_contours, -1, (0, 255, 0), 2)
    
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
   
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
   
    filled_result = img_rgb.copy()
    
    
    object_data = []
    
    for i, contour in enumerate(smoothed_contours):
        
        color = (
            (i * 50) % 255,
            (i * 80) % 255,
            (i * 110) % 255
        )
        cv2.drawContours(filled_result, [contour], 0, color, -1)
        
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
       
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        
        center = rect[0]
        width, height = rect[1]
        angle = rect[2]
        
       
        if width < height:
            width, height = height, width
            angle += 90
        
        
        angle = angle % 180
        
       
        aspect_ratio = width / height if height > 0 else float('inf')
        
        
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
       
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
        
       
        object_data.append({
            'object_id': i + 1,
            'area': area,
            'perimeter': perimeter,
            'center_x': center[0],
            'center_y': center[1],
            'width': width,
            'height': height,
            'orientation_angle': angle,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'centroid_x': cx,
            'centroid_y': cy
        })
    
    filled_output_path = output_path.replace('.jpg', '_filled.jpg').replace('.png', '_filled.png')
    filled_result_bgr = cv2.cvtColor(filled_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filled_output_path, filled_result_bgr)
    
    return result, filled_result, object_data

def watershed_refinement(image_path, output_path):
    """
    Advanced watershed-based segmentation for overlapping objects
    """
   
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
   
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
   
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
   
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    dist_threshold = 0.6 * dist_transform.max()
    _, sure_fg = cv2.threshold(dist_transform, dist_threshold, 255, 0)
    
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    
    _, markers = cv2.connectedComponents(sure_fg)
    
    
    markers = markers + 1
    
   
    markers[unknown == 255] = 0
    
    
    markers = cv2.watershed(img, markers)
    
   
    result = img_rgb.copy()
    
    
    result[markers == -1] = [255, 0, 0]
    
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
   
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    
    colored_regions = np.zeros_like(img_rgb)
    
    
    object_data = []
    
    
    for label in np.unique(markers):
        if label <= 0:  
            continue
            
       
        label_mask = np.zeros_like(gray, dtype=np.uint8)
        label_mask[markers == label] = 255
        
        
        label_contours, _ = cv2.findContours(
            label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(label_contours) == 0:
            continue
            
       
        contour = max(label_contours, key=cv2.contourArea)
        
       
        color = (
            (label * 50) % 255,
            (label * 80) % 255,
            (label * 110) % 255
        )
        
        colored_regions[markers == label] = color
        
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        
        center = rect[0]
        width, height = rect[1]
        angle = rect[2]
        
        
        if width < height:
            width, height = height, width
            angle += 90
        
        
        angle = angle % 180
        
        
        aspect_ratio = width / height if height > 0 else float('inf')
        
        
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
        
        
        object_data.append({
            'object_id': int(label),
            'area': area,
            'perimeter': perimeter,
            'center_x': center[0],
            'center_y': center[1],
            'width': width,
            'height': height,
            'orientation_angle': angle,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'centroid_x': cx,
            'centroid_y': cy
        })
    
    
    colored_output_path = output_path.replace('.jpg', '_regions.jpg').replace('.png', '_regions.png')
    colored_regions_bgr = cv2.cvtColor(colored_regions, cv2.COLOR_RGB2BGR)
    cv2.imwrite(colored_output_path, colored_regions_bgr)
    
    return result, colored_regions, object_data

def color_based_segmentation(image_path, output_path, color_ranges=None):
    """
    Color-based segmentation for detecting objects with specific colors
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    output_path : str
        Path to save output image
    color_ranges : list of dicts, optional
        List of color ranges to detect, each with 'name', 'lower', and 'upper' keys
        If None, default blue and red ranges will be used
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    
    if color_ranges is None:
        color_ranges = [
            {
                'name': 'blue',
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            {
                'name': 'red1',
                'lower': np.array([0, 50, 50]),
                'upper': np.array([10, 255, 255])
            },
            {
                'name': 'red2',
                'lower': np.array([170, 50, 50]),
                'upper': np.array([180, 255, 255])
            },
            {
                'name': 'green',
                'lower': np.array([40, 50, 50]),
                'upper': np.array([80, 255, 255])
            },
            {
                'name': 'yellow',
                'lower': np.array([20, 100, 100]),
                'upper': np.array([40, 255, 255])
            }
        ]
    
    
    masks = {}
    for color_range in color_ranges:
        mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        masks[color_range['name']] = mask
    
    
    combined_mask = np.zeros_like(masks[list(masks.keys())[0]])
    for mask_name, mask in masks.items():
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    
    contours, _ = cv2.findContours(
        refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    
    min_area = 100  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    
    result = img_rgb.copy()
    
    
    object_data = []
    
    
    for i, contour in enumerate(filtered_contours):
       
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
      
        obj_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(obj_mask, [contour], 0, 255, -1)
        
        
        color_scores = {}
        for color_name, color_mask in masks.items():
            overlap = cv2.bitwise_and(obj_mask, color_mask)
            overlap_score = np.sum(overlap) / 255
            color_scores[color_name] = overlap_score
        
       
        dominant_color = max(color_scores.items(), key=lambda x: x[1])[0]
        if dominant_color.startswith('red'):
            dominant_color = 'red'
        
        if circularity > 0.8:
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            
            
            object_data.append({
                'object_id': i + 1,
                'shape': 'circle',
                'color': dominant_color,
                'area': area,
                'perimeter': perimeter,
                'center_x': x,
                'center_y': y,
                'radius': radius,
                'diameter': radius * 2,
                'circularity': circularity,
                'orientation_angle': 0 
            })
        else:
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
            
           
            center = rect[0]
            width, height = rect[1]
            angle = rect[2]
            
            
            if width < height:
                width, height = height, width
                angle += 90
            
            
            angle = angle % 180
            
           
            aspect_ratio = width / height if height > 0 else float('inf')
            
          
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
            
            
            object_data.append({
                'object_id': i + 1,
                'shape': 'rectangle' if aspect_ratio > 1.2 else 'polygon',
                'color': dominant_color,
                'area': area,
                'perimeter': perimeter,
                'center_x': center[0],
                'center_y': center[1],
                'width': width,
                'height': height,
                'orientation_angle': angle,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'centroid_x': cx,
                'centroid_y': cy
            })
    
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    
    filled_result = img_rgb.copy()
    for i, obj in enumerate(object_data):
        
        color = (
            (i * 50) % 255,
            (i * 80) % 255,
            (i * 110) % 255
        )
        
        if obj['shape'] == 'circle':
            cv2.circle(filled_result, (int(obj['center_x']), int(obj['center_y'])), 
                      int(obj['radius']), color, -1)
        else:
           
            this_contour = filtered_contours[i]
            cv2.drawContours(filled_result, [this_contour], 0, color, -1)
    
    filled_output_path = output_path.replace('.jpg', '_filled.jpg').replace('.png', '_filled.png')
    filled_result_bgr = cv2.cvtColor(filled_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filled_output_path, filled_result_bgr)
    
    
    mask_output_path = output_path.replace('.jpg', '_mask.jpg').replace('.png', '_mask.png')
    cv2.imwrite(mask_output_path, refined_mask)
    
    return result, filled_result, object_data

def save_to_csv(object_data, output_path):
    """
    Save object metrics to CSV file
    
    Parameters:
    -----------
    object_data : list of dict
        List of dictionaries containing object metrics
    output_path : str
        Path to save CSV file
    """
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    
    all_fields = set()
    for obj in object_data:
        all_fields.update(obj.keys())
    
    
    fields = sorted(list(all_fields))
    
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(object_data)
    
    print(f"Saved object data to {output_path}")

def process_image(image_path, output_dir="./output", methods=None):
    """
    Process an image using multiple boundary detection methods and generate comparison
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    output_dir : str
        Directory to save output images
    methods : list of str, optional
        List of methods to apply. Default: ['basic', 'watershed', 'color']
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if methods is None:
        methods = ['basic', 'watershed', 'color']
    
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    global gray
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    results = {}
    
    if 'basic' in methods:
        basic_output_path = os.path.join(output_dir, f"{img_name}_basic.jpg")
        basic_result, basic_filled, basic_data = refine_boundaries(image_path, basic_output_path)
        results['basic'] = {
            'result': basic_result,
            'filled': basic_filled,
            'data': basic_data,
            'path': basic_output_path
        }
        
        basic_csv_path = os.path.join(output_dir, f"{img_name}_basic_metrics.csv")
        save_to_csv(basic_data, basic_csv_path)
        print(f"Basic method: Detected {len(basic_data)} objects")
    
    if 'watershed' in methods:
        watershed_output_path = os.path.join(output_dir, f"{img_name}_watershed.jpg")
        watershed_result, watershed_regions, watershed_data = watershed_refinement(image_path, watershed_output_path)
        results['watershed'] = {
            'result': watershed_result,
            'regions': watershed_regions,
            'data': watershed_data,
            'path': watershed_output_path
        }
        watershed_csv_path = os.path.join(output_dir, f"{img_name}_watershed_metrics.csv")
        save_to_csv(watershed_data, watershed_csv_path)
        print(f"Watershed method: Detected {len(watershed_data)} objects")
    
    if 'color' in methods:
        color_output_path = os.path.join(output_dir, f"{img_name}_color.jpg")
        color_result, color_filled, color_data = color_based_segmentation(image_path, color_output_path)
        results['color'] = {
            'result': color_result,
            'filled': color_filled,
            'data': color_data,
            'path': color_output_path
        }
        color_csv_path = os.path.join(output_dir, f"{img_name}_color_metrics.csv")
        save_to_csv(color_data, color_csv_path)
        print(f"Color method: Detected {len(color_data)} objects")
        
        print("Detected objects by color method:")
        for i, obj in enumerate(color_data):
            if 'shape' in obj:
                shape = obj['shape']
                if shape == 'circle':
                    print(f"  Object {i+1}: Circle with radius {obj['radius']:.1f}, " +
                          f"area {obj['area']:.1f}, color: {obj.get('color', 'unknown')}")
                else:
                    print(f"  Object {i+1}: {shape.capitalize()} with area {obj['area']:.1f}, " +
                          f"angle {obj.get('orientation_angle', 0):.1f}°, color: {obj.get('color', 'unknown')}")
    n_methods = len(results)
    if n_methods > 0:
        plt.figure(figsize=(15, 5 * (n_methods + 1) // 2))
        
        plt.subplot(n_methods + 1, 2, 1)
        plt.title("Original Image")
        plt.imshow(original_rgb)
        plt.axis('off')
        
        i = 2
        for method_name, method_results in results.items():
            plt.subplot(n_methods + 1, 2, i)
            plt.title(f"{method_name.capitalize()} Method ({len(method_results['data'])} objects)")
            plt.imshow(method_results['result'])
            plt.axis('off')
            i += 1
            
            if 'filled' in method_results:
                plt.subplot(n_methods + 1, 2, i)
                plt.title(f"{method_name.capitalize()} Filled")
                plt.imshow(method_results['filled'])
                plt.axis('off')
                i += 1
            elif 'regions' in method_results:
                plt.subplot(n_methods + 1, 2, i)
                plt.title(f"{method_name.capitalize()} Regions")
                plt.imshow(method_results['regions'])
                plt.axis('off')
                i += 1
        
        comparison_path = os.path.join(output_dir, f"{img_name}_comparison.jpg")
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300)
        print(f"Comparison saved to {comparison_path}")
    
    return results

if __name__ == "__main__":
    input_path = r"C:\Users\ytewa\Downloads\es3 (1).jpg"
    
    output_dir = os.path.dirname(input_path)
    
    try:
        results = process_image(input_path, output_dir, ['basic', 'watershed', 'color'])
        print(f"Processing complete. Results saved to {output_dir}")
        print(f"Check {output_dir} for CSV files with area and orientation metrics")
    except Exception as e:
        print(f"Error processing image: {e}")
