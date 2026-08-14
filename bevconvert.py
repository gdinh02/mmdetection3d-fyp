import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
import numpy as np

import cv2
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=5, min_samples=3)

from skimage.morphology import skeletonize

def setup_visualization():
    """
    Initializes the matplotlib window once.
    Call this BEFORE your video processing loop.
    """
    plt.ion()  # Turn on interactive mode for live updating
    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(16, 8))
    fig.canvas.manager.set_window_title("3D Perception BEV")
    plt.tight_layout()
    return fig, ax_img, ax_bev

def generate_binary_bev_map(bboxes_3d, scores_3d, score_thresh, width=100, height=150, buffer_meters=0.5):
    """Draws bounding boxes onto a 2D NumPy array with an oriented cross-shaped buffer."""
    binary_map = np.zeros((height, width), dtype=np.float32)
    
    # Grid parameters based on previous -20 to 20 X, and 0 to 50 Z
    x_range, z_range = 40.0, 50.0
    x_res = width / x_range
    z_res = height / z_range
    
    for box, score in zip(bboxes_3d, scores_3d):
        if score < score_thresh:
            continue
            
        x, y, z, dx, dy, dz, roll, pitch, yaw = box[:9]
        
        # Convert center coordinates
        px_x = int((x + 20) * x_res)  
        px_z = int(z * z_res)
        
        # To create an oriented cross, we draw TWO intersecting rectangles per vehicle.
        # We multiply the buffer by 2 because it applies to both sides of the center.
        
        # 1. Horizontal bar of the cross (Dilated Width, Original Length)
        w1_buffered = dx + (buffer_meters * 2)
        px_w1 = int(w1_buffered * x_res)
        px_l1 = int(dz * z_res)
        rect1 = cv2.boxPoints(((px_x, px_z), (px_w1, px_l1), np.degrees(-roll)))
        
        # 2. Vertical bar of the cross (Original Width, Dilated Length)
        w2_original = dx
        l2_buffered = dz + (buffer_meters * 2)
        px_w2 = int(w2_original * x_res)
        px_l2 = int(l2_buffered * z_res)
        rect2 = cv2.boxPoints(((px_x, px_z), (px_w2, px_l2), np.degrees(-roll)))
        
        # Draw both polygons onto the map
        cv2.fillPoly(binary_map, [np.int32(rect1)], color=1.0)
        cv2.fillPoly(binary_map, [np.int32(rect2)], color=1.0)
        
    return binary_map

def align_multi_frame_history(curr_map, history_buffer, weights):
    """
    Aligns a buffer of previous raw maps to the current map and computes a weighted sum.
    
    Args:
        curr_map: The current frame's raw binary map.
        history_buffer: A list/deque of previous raw maps (index 0 is t-1, index 1 is t-2, etc.).
        weights: A list of weights for [current, t-1, t-2, ...]. Must sum to 1.0.
    """
    # Start the final map with the current frame's weighted values
    final_map = (curr_map * weights[0]).astype(np.float32)
    
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)

    blur_size = (21, 21)
    curr_map_blur = cv2.GaussianBlur(curr_map, blur_size, 0)
    
    for i, prev_map in enumerate(history_buffer):
        weight = weights[i + 1]
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        prev_map_blur = cv2.GaussianBlur(prev_map, blur_size, 0)
        
        try:
            # Align the historical raw map to the current raw map
            _, warp_matrix = cv2.findTransformECC(
                templateImage=curr_map_blur,
                inputImage=prev_map_blur,
                warpMatrix=warp_matrix,
                motionType=warp_mode,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=1
            )
            
            aligned_prev = cv2.warpAffine(
                prev_map, 
                warp_matrix, 
                (curr_map.shape[1], curr_map.shape[0]), 
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )

            print(warp_matrix[:, 2])
        except cv2.error:
            # Fallback if ECC fails (e.g., zero overlap)
            print("Align failed")
            aligned_prev = prev_map 
            
        # Add the weighted historical frame to the final map
        final_map += (aligned_prev * weight)
        
    return final_map


def smooth_binary_map(bev_map, apply_closing=True, apply_blur=True, buffer_meters=0.5, resolution=0.2):
    """
    Applies spatial and morphological smoothing to a BEV map.
    
    Args:
        bev_map: The 2D numpy array (binary or blended map).
        apply_closing: If True, connects nearby objects and fills holes.
        apply_blur: If True, softens the edges into a gradient costmap.
        buffer_meters: How much to expand the obstacles for safety.
        resolution: Meters per pixel (e.g., 40m / 200px = 0.2m/px).
    """
    smoothed_map = bev_map.copy()

    # 2. Spatial Smoothing (Gaussian Blur)
    if apply_blur:
        # A large kernel creates a wider, smoother gradient around objects
        blur_kernel = (11, 11) 
        smoothed_map = cv2.GaussianBlur(smoothed_map, blur_kernel, sigmaX=3, sigmaY=3)
        
    return smoothed_map

def update_visualization(fig, ax_img, ax_bev, image_path, bboxes_3d, scores_3d, labels_3d, blended_map=None, score_thresh=0.3, timeout=0.1):
    """
    Clears the axes and redraws the new frame data.
    """
    ax_img.clear()
    ax_bev.clear()
    
    # ==========================================
    # 1. Plot the Raw Camera Image
    # ==========================================
    try:
        img = mpimg.imread(image_path)
        ax_img.imshow(img)
        ax_img.set_title("Front Camera Image")
        ax_img.axis('off')
    except FileNotFoundError:
        ax_img.text(0.5, 0.5, 'Image not found.', ha='center', va='center')
        ax_img.axis('off')

    # ==========================================
    # 2. Render the Blended Binary Map
    # ==========================================
    if blended_map is not None:
        ax_bev.imshow(
            blended_map, 
            extent=[-20, 20, 0, 50], # Ensure this matches your grid limits!
            origin='lower', 
            cmap='gray',             # Grayscale works best for binary maps
            alpha=0.6,               # Slight transparency
            zorder=1
        )

    # ==========================================
    # 3. Plot the BEV Bounding Boxes & Arrows
    # ==========================================

    # 1. Pre-filter the lists to ensure exact index alignment with DBSCAN outputs
    valid_objects = [(box, score, label) for box, score, label in zip(bboxes_3d, scores_3d, labels_3d) if score >= score_thresh]

    # Only attempt clustering if there are valid objects to avoid empty array errors
    if valid_objects:
        
        # Extract X and Z for DBSCAN
        obj_locs = [[obj[0][0], obj[0][2]] for obj in valid_objects]
        
        # Capture the output labels
        cluster_labels = dbscan.fit_predict(obj_locs)
        
        # Load a Matplotlib colormap (tab10 provides 10 distinct, highly-contrasting colors)
        cmap = plt.colormaps['tab10']

        # Iterate through the filtered objects and their corresponding DBSCAN label simultaneously
        for (box, score, label), cluster_id in zip(valid_objects, cluster_labels):
            
            x, y, z, dx, dy, dz, roll, pitch, yaw = box[:9]
            
            bev_x = x
            bev_y = z
            bev_w = dx  
            bev_l = dz  
            
            # ------------------------------------------
            # Draw DBSCAN Cluster Circle
            # ------------------------------------------
            if cluster_id == -1:
                # Noise points get a dotted gray circle
                cluster_color = 'gray'
                linestyle = ':'
            else:
                # Valid clusters loop through the 10 colormap colors based on their ID
                cluster_color = cmap(cluster_id % 10)
                linestyle = '--'
                
            cluster_circle = plt.Circle(
                (bev_x, bev_y),
                radius=1.5,             # Radius of the circle in meters
                edgecolor=cluster_color,
                facecolor='none',       # Keep the center transparent
                linewidth=2.0,
                linestyle=linestyle,
                zorder=2                # Render underneath the bounding box
            )
            ax_bev.add_patch(cluster_circle)
            
            # # ------------------------------------------
            # # Draw Rectangle
            # # ------------------------------------------
            rect_x = bev_x - bev_w / 2
            rect_y = bev_y - bev_l / 2
            
            rect = plt.Rectangle(
                (rect_x, rect_y),
                bev_w,
                bev_l,
                linewidth=1.5,
                edgecolor='orange' if label == 7 else 'blue',
                facecolor='orange' if label == 7 else 'blue',
                alpha=0.4,
                zorder=3
            )
            
            # Apply box rotation
            transform = Affine2D().rotate_around(bev_x, bev_y, -roll) + ax_bev.transData
            rect.set_transform(transform)
            ax_bev.add_patch(rect)

            # ------------------------------------------
            # Draw Extended Edge Lines (Doubled Length)
            # ------------------------------------------
            
            # # Calculate the 50% extensions for the unrotated bounding box
            # x_ext_start = rect_x - (bev_w * 0.5)
            # x_ext_end   = rect_x + bev_w + (bev_w * 0.5)
            
            # y_ext_start = rect_y - (bev_l * 0.5)
            # y_ext_end   = rect_y + bev_l + (bev_l * 0.5)
            
            # line_color = 'red' # Change to whatever color you prefer
            # line_width = 1.0
            
            # # Left edge extended (Vertical)
            # ax_bev.plot([rect_x, rect_x], [y_ext_start, y_ext_end], 
            #             color=line_color, linewidth=line_width, transform=transform, zorder=4)
            
            # # Right edge extended (Vertical)
            # ax_bev.plot([rect_x + bev_w, rect_x + bev_w], [y_ext_start, y_ext_end], 
            #             color=line_color, linewidth=line_width, transform=transform, zorder=4)
            
            # # Bottom edge extended (Horizontal)
            # ax_bev.plot([x_ext_start, x_ext_end], [rect_y, rect_y], 
            #             color=line_color, linewidth=line_width, transform=transform, zorder=4)
            
            # # Top edge extended (Horizontal)
            # ax_bev.plot([x_ext_start, x_ext_end], [rect_y + bev_l, rect_y + bev_l], 
            #             color=line_color, linewidth=line_width, transform=transform, zorder=4)
            
            # ------------------------------------------
            # Draw Directional Arrow
            # ------------------------------------------
            arrow_len = bev_w / 2.0 
            
            arrow_dx = arrow_len * np.cos(-roll)
            arrow_dy = arrow_len * np.sin(-roll)
            
            ax_bev.arrow(
                bev_x, bev_y,        
                arrow_dx, arrow_dy,  
                width=0.1,           
                head_width=0.8,      
                head_length=0.8,     
                fc='white',          
                ec='black',          
                linewidth=0.5,
                zorder=10,           
                length_includes_head=True
            )

            skeleton = skeletonize(blended_map)

            skeleton_masked = np.ma.masked_where(~skeleton, skeleton)

            ax_bev.imshow(
                skeleton_masked, 
                extent=[-20, 20, 0, 50], # Must use the exact same extent
                origin='lower', 
                cmap='autumn',           # 'autumn' maps the value '1' to solid red
                alpha=1.0,               # Keep skeleton fully opaque
                zorder=2,                # Render on top of the blended map
                interpolation='none'     # Prevents matplotlib from blurring the 1-pixel thin line
            )
        
    # Plot Ego Vehicle
    ax_bev.scatter([0], [0], color='green', s=100, label='Ego Vehicle (Camera)', zorder=5)
    
    # Re-apply Graph styling
    ax_bev.set_xlim(-20, 20)
    ax_bev.set_ylim(0, 50)
    ax_bev.set_xlabel('Lateral Distance X (m)')
    ax_bev.set_ylabel('Forward Distance Z (m)')
    ax_bev.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_bev.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax_bev.grid(True, which='both', linestyle=':', alpha=0.5)
    ax_bev.set_title("Bird's-Eye-View (BEV) Map")
    ax_bev.legend(loc='lower right')
    ax_bev.set_aspect('equal', adjustable='box')

    if timeout:
        plt.pause(timeout)
    else:
        fig.canvas.draw()
        fig.canvas.flush_events()
        while not plt.waitforbuttonpress():
            pass