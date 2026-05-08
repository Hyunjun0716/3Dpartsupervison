"""
Visualize Point Clouds from Random Custom Captions using Open3D
This script:
1. Loads a custom caption list
2. Generates point clouds from random captions
3. Renders them as 2D images using Open3D's PointCloud renderer
"""

import os
import argparse
import torch
import numpy as np
import random
import time
from tqdm.auto import tqdm
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from models.vae_flow import FlowVAE
from models.clip_encoder import FrozenCLIPTextEmbedder


def normalize_point_cloud(pc, mode='shape_unit'):
    """
    Normalize point cloud

    Args:
        pc: (N, 3) point cloud tensor or numpy array
        mode: normalization mode
    """
    is_tensor = torch.is_tensor(pc)
    if is_tensor:
        if mode == 'shape_unit':
            shift = pc.mean(dim=0, keepdim=True)
            scale = pc.flatten().std()
        elif mode == 'shape_bbox':
            pc_max = pc.max(dim=0, keepdim=True)[0]
            pc_min = pc.min(dim=0, keepdim=True)[0]
            shift = (pc_min + pc_max) / 2
            scale = (pc_max - pc_min).max() / 2
        else:
            return pc
        pc = (pc - shift) / scale
    else:
        # numpy version
        if mode == 'shape_unit':
            shift = pc.mean(axis=0, keepdims=True)
            scale = pc.flatten().std()
        elif mode == 'shape_bbox':
            pc_max = pc.max(axis=0, keepdims=True)
            pc_min = pc.min(axis=0, keepdims=True)
            shift = (pc_min + pc_max) / 2
            scale = (pc_max - pc_min).max() / 2
        else:
            return pc
        pc = (pc - shift) / scale

    return pc


def render_point_cloud_to_image(points, save_path, window_name="Point Cloud",
                                width=800, height=600, point_size=2.0,
                                view_angle='default', bg_color=[1, 1, 1]):
    """
    Render point cloud to 2D image using Open3D's offscreen rendering

    Args:
        points: (N, 3) numpy array of point coordinates
        save_path: path to save the rendered image
        window_name: window title
        width: image width
        height: image height
        point_size: size of rendered points
        view_angle: 'default', 'front', 'side', 'top', or custom angles (elev, azim)
        bg_color: background color [R, G, B] in range [0, 1]
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals for better shading
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Color the point cloud (gradient based on Z coordinate for depth perception)
    z_coords = points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)

    # Create color gradient (blue to red)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_normalized  # Red channel
    colors[:, 2] = 1 - z_normalized  # Blue channel
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray(bg_color)
    render_option.show_coordinate_frame = False
    render_option.light_on = True

    # Set camera view
    view_control = vis.get_view_control()

    cam_distance = 3.0

    if view_angle == 'front':
        view_control.set_lookat([0, 0, 0])
        view_control.set_front([0, 0, 1])
        view_control.set_up([0, 1, 0])
    elif view_angle == 'side':
        view_control.set_lookat([0, 0, 0])
        view_control.set_front([-1, 0, 0])
        view_control.set_up([0, 1, 0])
    elif view_angle == 'top':
        view_control.set_lookat([0, 0, 0])
        view_control.set_front([0, 1, 0])
        view_control.set_up([0, 0, -1])
    elif isinstance(view_angle, tuple) and len(view_angle) == 2:
        # Custom angle: (elevation, azimuth) in degrees
        elev, azim = view_angle
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)

        # Convert spherical to Cartesian
        x = cam_distance * np.cos(elev_rad) * np.sin(azim_rad)
        y = cam_distance * np.sin(elev_rad)
        z = cam_distance * np.cos(elev_rad) * np.cos(azim_rad)

        view_control.set_lookat([0, 0, 0])
        view_control.set_front([-x, -y, -z])
        view_control.set_up([0, 1, 0])
    else:
        # Default: slight angle for better 3D perception
        elev, azim = 20, 45
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)

        x = cam_distance * np.cos(elev_rad) * np.sin(azim_rad)
        y = cam_distance * np.sin(elev_rad)
        z = cam_distance * np.cos(elev_rad) * np.cos(azim_rad)

        view_control.set_lookat([0, 0, 0])
        view_control.set_front([-x, -y, -z])
        view_control.set_up([0, 1, 0])

    view_control.set_zoom(0.5)

    # Update geometry and render
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Capture and save image
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()

    print(f"[INFO] Saved rendered image: {save_path}")


def render_point_cloud_grid(point_clouds, captions, save_path,
                            num_rotations=5, rotation_axis='y',
                            img_width=256, img_height=256,
                            point_size=5.0, bg_color=[1, 1, 1]):
    """
    Render multiple point clouds in a grid layout with rotation sequence

    Args:
        point_clouds: list of (N, 3) numpy arrays
        captions: list of caption strings
        save_path: path to save the grid image
        num_rotations: number of rotation angles per sample
        rotation_axis: 'y' for horizontal rotation, 'x' for vertical
        img_width: width of each cell
        img_height: height of each cell
        point_size: size of rendered points
        bg_color: background color [R, G, B]
    """
    num_samples = len(point_clouds)

    # Calculate rotation angles (0 to 360 degrees)
    angles = np.linspace(0, 360, num_rotations, endpoint=False)

    print(f"[INFO] Creating grid: {num_samples} samples x {num_rotations} rotations")

    # Create temporary directory for individual renders
    import tempfile
    temp_dir = tempfile.mkdtemp()

    # Render each point cloud at each rotation angle
    all_images = []

    for i, (pc, caption) in enumerate(zip(point_clouds, captions)):
        row_images = []

        for j, angle in enumerate(angles):
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)

            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

            # Color based on Z coordinate
            z_coords = pc[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)

            colors = np.zeros((len(pc), 3))
            colors[:, 0] = z_normalized  # Red
            colors[:, 2] = 1 - z_normalized  # Blue
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{caption} - {angle:.0f}°",
                width=img_width,
                height=img_height,
                visible=False
            )
            vis.add_geometry(pcd)

            # Set render options
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            render_option.background_color = np.asarray(bg_color, dtype=np.float64)
            render_option.show_coordinate_frame = False
            render_option.point_show_normal = False

            # Set camera view with rotation
            view_control = vis.get_view_control()

            # Calculate camera position based on rotation
            if rotation_axis == 'y':
                # Rotate around Y axis (horizontal rotation)
                azim = angle
                elev = 20
            else:
                # Rotate around X axis (vertical rotation)
                azim = 45
                elev = angle

            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)

            # Camera looks at origin from distance
            cam_distance = 3.0
            x = cam_distance * np.cos(elev_rad) * np.sin(azim_rad)
            y = cam_distance * np.sin(elev_rad)
            z = cam_distance * np.cos(elev_rad) * np.cos(azim_rad)

            view_control.set_lookat([0, 0, 0])  # Look at center
            view_control.set_front([-x, -y, -z])  # Camera direction
            view_control.set_up([0, 1, 0])  # Up direction
            view_control.set_zoom(0.5)  # Adjust zoom to fit object

            # Update and render
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Save to temporary file
            temp_path = os.path.join(temp_dir, f"sample_{i}_angle_{j}.png")
            vis.capture_screen_image(temp_path, do_render=True)
            vis.destroy_window()

            # Load image
            img = Image.open(temp_path)
            row_images.append(img)

        all_images.append(row_images)

    # Create grid image
    grid_width = img_width * num_rotations
    grid_height = img_height * num_samples

    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')

    for i, row_images in enumerate(all_images):
        for j, img in enumerate(row_images):
            x_offset = j * img_width
            y_offset = i * img_height
            grid_img.paste(img, (x_offset, y_offset))

    # Save grid image
    grid_img.save(save_path, dpi=(150, 150))

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)

    print(f"[SUCCESS] Saved grid image: {save_path}")
    print(f"           Size: {grid_width} x {grid_height} ({num_samples} rows x {num_rotations} columns)")


def render_denoising_process(trajectories, captions, save_path, num_steps=5,
                             figsize_per_cell=(3, 3), point_size=1.0,
                             color_scheme='skyblue', angle=30):
    """
    Visualize the denoising process from noise to final point cloud

    Args:
        trajectories: dict mapping timestep -> point cloud tensor, for each caption
                     List of dicts, one per caption
        captions: list of caption strings
        save_path: path to save the visualization
        num_steps: number of intermediate steps to visualize (including start and end)
        figsize_per_cell: (width, height) size per cell in inches
        point_size: size of rendered points
        color_scheme: color scheme for points
        angle: viewing angle in degrees
    """
    num_samples = len(trajectories)

    print(f"[INFO] Creating denoising process visualization: {num_samples} samples x {num_steps} steps")

    # Create figure with subplots
    fig_width = figsize_per_cell[0] * num_steps
    fig_height = figsize_per_cell[1] * num_samples
    fig = plt.figure(figsize=(fig_width, fig_height))

    for i, (traj, caption) in enumerate(zip(trajectories, captions)):
        # Get all timesteps and select num_steps evenly spaced ones
        timesteps = sorted(traj.keys(), reverse=True)  # From T (noise) to 0 (clean)

        # Select evenly spaced timesteps
        if len(timesteps) >= num_steps:
            step_indices = np.linspace(0, len(timesteps)-1, num_steps, dtype=int)
            selected_timesteps = [timesteps[idx] for idx in step_indices]
        else:
            selected_timesteps = timesteps

        for j, t in enumerate(selected_timesteps):
            # Get point cloud at timestep t
            pc_tensor = traj[t]
            if torch.is_tensor(pc_tensor):
                pc = pc_tensor.squeeze(0).cpu().numpy()  # (N, 3)
            else:
                pc = pc_tensor

            # Create subplot
            subplot_idx = i * num_steps + j + 1
            ax = fig.add_subplot(num_samples, num_steps, subplot_idx, projection='3d')

            # Set colors based on color scheme
            if color_scheme == 'skyblue':
                colors = np.tile([0.529, 0.808, 0.922], (len(pc), 1))
            elif color_scheme == 'gradient_blue':
                z_coords = pc[:, 2]
                z_min, z_max = z_coords.min(), z_coords.max()
                z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)
                colors = np.zeros((len(pc), 3))
                colors[:, 0] = 0.3 + 0.3 * (1 - z_normalized)
                colors[:, 1] = 0.6 + 0.3 * (1 - z_normalized)
                colors[:, 2] = 0.8 + 0.2 * z_normalized
            elif color_scheme == 'depth':
                z_coords = pc[:, 2]
                z_min, z_max = z_coords.min(), z_coords.max()
                z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)
                colors = np.zeros((len(pc), 3))
                colors[:, 0] = z_normalized
                colors[:, 2] = 1 - z_normalized
            elif color_scheme == 'height':
                y_coords = pc[:, 1]
                y_min, y_max = y_coords.min(), y_coords.max()
                y_normalized = (y_coords - y_min) / (y_max - y_min + 1e-8)
                colors = np.zeros((len(pc), 3))
                colors[:, 1] = 0.3 + 0.7 * y_normalized
            else:
                colors = np.tile([0.7, 0.7, 0.7], (len(pc), 1))

            # Plot point cloud
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                      c=colors, s=point_size, alpha=0.8, marker='o', edgecolors='none')

            # Set viewing angle
            ax.view_init(elev=20, azim=angle)

            # Set equal aspect ratio
            max_range = np.abs(pc).max()
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

            # Hide all axes lines and panes
            ax.set_axis_off()

            # Add title - show timestep and caption on first step
            if j == 0:
                truncated_caption = caption[:30] + "..." if len(caption) > 30 else caption
                ax.set_title(f"{truncated_caption}\nt={t}", fontsize=8, pad=2)
            else:
                ax.set_title(f"t={t}", fontsize=8, pad=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[SUCCESS] Saved denoising process visualization: {save_path}")
    print(f"           Grid: {num_samples} rows x {num_steps} steps")


def render_point_cloud_grid_3d(point_clouds, captions, save_path,
                               num_rotations=5, rotation_axis='y',
                               figsize_per_cell=(3, 3), point_size=1.0, color_scheme='depth',
                               custom_angles=None):
    """
    Render multiple point clouds in a 3D grid layout using matplotlib

    Args:
        point_clouds: list of (N, 3) numpy arrays
        captions: list of caption strings
        save_path: path to save the grid image
        num_rotations: number of rotation angles per sample (ignored if custom_angles is set)
        rotation_axis: 'y' for horizontal rotation, 'x' for vertical
        figsize_per_cell: (width, height) size per cell in inches
        point_size: size of rendered points
        custom_angles: list of specific angles in degrees (overrides num_rotations)
    """
    num_samples = len(point_clouds)

    # Calculate rotation angles
    if custom_angles is not None:
        angles = np.array(custom_angles)
        num_rotations = len(angles)
    else:
        # Auto-generate evenly spaced angles (0 to 360 degrees)
        angles = np.linspace(0, 360, num_rotations, endpoint=False)

    print(f"[INFO] Creating 3D grid: {num_samples} samples x {num_rotations} rotations")

    # Create figure with subplots
    fig_width = figsize_per_cell[0] * num_rotations
    fig_height = figsize_per_cell[1] * num_samples
    fig = plt.figure(figsize=(fig_width, fig_height))

    for i, (pc, caption) in enumerate(zip(point_clouds, captions)):
        for j, angle in enumerate(angles):
            # Create subplot
            subplot_idx = i * num_rotations + j + 1
            ax = fig.add_subplot(num_samples, num_rotations, subplot_idx, projection='3d')

            # Set colors based on color scheme
            if color_scheme == 'skyblue':
                # Solid sky blue color
                colors = np.tile([0.529, 0.808, 0.922], (len(pc), 1))  # RGB for sky blue
            elif color_scheme == 'gradient_blue':
                # Light blue to dark blue gradient based on depth
                z_coords = pc[:, 2]
                z_min, z_max = z_coords.min(), z_coords.max()
                z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)

                # Gradient from light blue to dark blue
                colors = np.zeros((len(pc), 3))
                colors[:, 0] = 0.3 + 0.3 * (1 - z_normalized)  # R
                colors[:, 1] = 0.6 + 0.3 * (1 - z_normalized)  # G
                colors[:, 2] = 0.8 + 0.2 * z_normalized  # B
            elif color_scheme == 'depth':
                # Blue to red gradient based on depth (original)
                z_coords = pc[:, 2]
                z_min, z_max = z_coords.min(), z_coords.max()
                z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)

                colors = np.zeros((len(pc), 3))
                colors[:, 0] = z_normalized  # Red channel
                colors[:, 2] = 1 - z_normalized  # Blue channel
            elif color_scheme == 'height':
                # Green gradient based on Y coordinate
                y_coords = pc[:, 1]
                y_min, y_max = y_coords.min(), y_coords.max()
                y_normalized = (y_coords - y_min) / (y_max - y_min + 1e-8)

                colors = np.zeros((len(pc), 3))
                colors[:, 1] = 0.3 + 0.7 * y_normalized  # Green channel
            else:
                # Default: uniform gray
                colors = np.tile([0.7, 0.7, 0.7], (len(pc), 1))

            # Plot point cloud
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                      c=colors, s=point_size, alpha=0.8, marker='o', edgecolors='none')

            # Set viewing angle
            if rotation_axis == 'y':
                # Rotate around Y axis (horizontal rotation)
                azim = angle
                elev = 20
            else:
                # Rotate around X axis (vertical rotation)
                azim = 45
                elev = angle

            ax.view_init(elev=elev, azim=azim)

            # Set equal aspect ratio
            max_range = np.abs(pc).max()
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

            # Remove axes completely
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Hide all axes lines and panes
            ax.set_axis_off()

            # Add title on first column (optional - can be removed if not needed)
            if j == 0:
                truncated_caption = caption[:30] + "..." if len(caption) > 30 else caption
                ax.set_title(f"{truncated_caption}\n{angle:.0f}°", fontsize=8, pad=2)
            else:
                ax.set_title(f"{angle:.0f}°", fontsize=8, pad=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[SUCCESS] Saved 3D grid image: {save_path}")
    print(f"           Grid: {num_samples} rows x {num_rotations} columns")


def visualize_interactive_3d(point_clouds, captions):
    """
    Open interactive 3D viewer with Open3D for multiple point clouds

    Args:
        point_clouds: list of (N, 3) numpy arrays
        captions: list of caption strings
    """
    print("\n[INFO] Opening interactive 3D viewer...")
    print("[INFO] Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - Arrow keys: Navigate between samples")
    print("  - Q or ESC: Close viewer")
    print()

    current_idx = 0

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Sample {current_idx + 1}/{len(point_clouds)}: {captions[current_idx]}",
                     width=1024, height=768)

    # Create initial point cloud
    pcd = o3d.geometry.PointCloud()
    pc = point_clouds[current_idx]
    pcd.points = o3d.utility.Vector3dVector(pc)

    # Color based on Z coordinate
    z_coords = pc[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)

    colors = np.zeros((len(pc), 3))
    colors[:, 0] = z_normalized  # Red
    colors[:, 2] = 1 - z_normalized  # Blue
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals for better shading
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    vis.add_geometry(pcd)

    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.background_color = np.asarray([1, 1, 1])

    def update_point_cloud(idx):
        """Update displayed point cloud"""
        nonlocal current_idx
        current_idx = idx % len(point_clouds)

        pc = point_clouds[current_idx]
        pcd.points = o3d.utility.Vector3dVector(pc)

        # Update colors
        z_coords = pc[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-8)

        colors = np.zeros((len(pc), 3))
        colors[:, 0] = z_normalized
        colors[:, 2] = 1 - z_normalized
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        vis.update_geometry(pcd)
        vis.reset_view_point(True)

        # Update window title
        new_title = f"Sample {current_idx + 1}/{len(point_clouds)}: {captions[current_idx]}"
        print(f"[INFO] Showing: {new_title}")

        return False

    # Key callbacks for navigation
    def next_sample(vis_):
        update_point_cloud(current_idx + 1)
        return False

    def prev_sample(vis_):
        update_point_cloud(current_idx - 1)
        return False

    # Register key callbacks
    vis.register_key_callback(262, next_sample)  # Right arrow
    vis.register_key_callback(263, prev_sample)  # Left arrow
    vis.register_key_callback(ord('N'), next_sample)  # N key
    vis.register_key_callback(ord('P'), prev_sample)  # P key

    print(f"[INFO] Showing Sample 1/{len(point_clouds)}: {captions[0]}")
    print("[INFO] Press RIGHT ARROW or N for next, LEFT ARROW or P for previous")

    # Run visualizer
    vis.run()
    vis.destroy_window()

    print("[INFO] Interactive viewer closed")


def load_captions_from_file(file_path):
    """
    Load captions from a text file (one caption per line)

    Args:
        file_path: path to captions file

    Returns:
        list of caption strings
    """
    captions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                captions.append(line)
    return captions


def main(args):
    # Set seed for reproducibility (use random seed if None)
    if args.seed is None:
        args.seed = int(time.time())
        print(f"[INFO] No seed specified, using random seed: {args.seed}")
    else:
        print(f"[INFO] Using specified seed: {args.seed}")
    seed_all(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("RANDOM CAPTION POINT CLOUD VISUALIZATION")
    print("="*80)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    print(f"[INFO] Checkpoint loaded from iteration {ckpt.get('iteration', 'unknown')}")

    # Load CLIP text encoder
    use_text = ckpt['args'].use_text_condition if hasattr(ckpt['args'], 'use_text_condition') else False

    if not use_text:
        print("[ERROR] Model was trained without text conditioning!")
        print("[ERROR] This script requires a text-conditioned model.")
        return

    print("[INFO] Loading CLIP text encoder...")
    clip_model = ckpt['args'].clip_model if hasattr(ckpt['args'], 'clip_model') else 'openai/clip-vit-base-patch32'
    text_encoder = FrozenCLIPTextEmbedder(
        version=clip_model,
        device=args.device,
        return_sequence=True
    )
    text_encoder = text_encoder.to(args.device)
    text_encoder.eval()

    # Load model
    print("[INFO] Loading model...")
    if 'latent_flow_depth' in ckpt['args']:
        print("[INFO] Model type: FlowVAE")
        model = FlowVAE(ckpt['args'], tokenizer=text_encoder.tokenizer).to(args.device)
    else:
        print("[INFO] Model type: GaussianVAE")
        model = GaussianVAE(ckpt['args'], tokenizer=text_encoder.tokenizer).to(args.device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Load captions
    if args.captions_file:
        print(f"[INFO] Loading captions from: {args.captions_file}")
        all_captions = load_captions_from_file(args.captions_file)
        print(f"[INFO] Loaded {len(all_captions)} captions")
    elif args.custom_captions:
        print(f"[INFO] Using {len(args.custom_captions)} custom captions from command line")
        all_captions = args.custom_captions
    else:
        print("[ERROR] No captions provided!")
        print("[ERROR] Use --captions_file or --custom_captions")
        return

    if len(all_captions) == 0:
        print("[ERROR] Caption list is empty!")
        return

    # Select random captions
    if args.num_samples > len(all_captions):
        print(f"[WARNING] Requested {args.num_samples} samples but only {len(all_captions)} captions available")
        print(f"[WARNING] Using all {len(all_captions)} captions")
        selected_captions = all_captions
    else:
        if args.sample_random:
            print(f"[INFO] Randomly selecting {args.num_samples} captions")
            selected_captions = random.sample(all_captions, args.num_samples)
        else:
            print(f"[INFO] Using first {args.num_samples} captions")
            selected_captions = all_captions[:args.num_samples]

    print("\n[INFO] Selected captions:")
    for i, caption in enumerate(selected_captions, 1):
        print(f"  {i}. {caption}")
    print()

    # Save selected captions to file
    captions_log = os.path.join(args.save_dir, 'selected_captions.txt')
    with open(captions_log, 'w') as f:
        for i, caption in enumerate(selected_captions, 1):
            f.write(f"{i}. {caption}\n")
    print(f"[INFO] Saved caption list to: {captions_log}")

    # Generate point clouds
    print(f"\n[INFO] Generating {len(selected_captions)} point clouds...")

    generated_pcs = []
    trajectories = [] if args.visualize_process else None
    latent_codes = []  # Store z values

    with torch.no_grad():
        for i, caption in enumerate(tqdm(selected_captions, desc="Generating")):
            # Encode text
            text_emb = text_encoder([caption])

            # Sample random latent code
            latent_dim = ckpt['args'].latent_dim
            z = torch.randn(1, latent_dim).to(args.device)

            # Store z value
            latent_codes.append(z.cpu().numpy())

            # Print z statistics for this sample
            z_np = z.cpu().numpy().flatten()
            print(f"\n[Z Stats for sample {i+1}] Caption: '{caption}'")
            print(f"  Shape: {z.shape}")
            print(f"  Mean: {z_np.mean():.4f}, Std: {z_np.std():.4f}")
            print(f"  Min: {z_np.min():.4f}, Max: {z_np.max():.4f}")
            print(f"  First 5 values: {z_np[:5]}")

            # Generate point cloud (with trajectory if requested)
            if args.visualize_process:
                # For trajectory, we need to call diffusion directly
                # First get the context (latent code z from flow prior for FlowVAE)
                if hasattr(model, 'flow'):
                    # FlowVAE: w -> z via flow
                    context = model.flow(z, reverse=True).view(z.size(0), -1)
                else:
                    # GaussianVAE: z is context directly
                    context = z

                # Sample with trajectory for denoising visualization
                traj = model.diffusion.sample(
                    num_points=args.sample_num_points,
                    context=context,
                    text_emb=text_emb,
                    flexibility=args.flexibility,
                    ret_traj=True
                )
                # Get final result (t=0)
                generated_pc = traj[0].squeeze(0).cpu()  # (N, 3)

                # Normalize all trajectory steps
                if args.normalize != 'none':
                    traj_normalized = {}
                    for t, pc_t in traj.items():
                        pc_t_cpu = pc_t.squeeze(0).cpu()
                        pc_t_normalized = normalize_point_cloud(pc_t_cpu, mode=args.normalize)
                        traj_normalized[t] = pc_t_normalized
                    trajectories.append(traj_normalized)
                    generated_pc = traj_normalized[0]
                else:
                    trajectories.append({t: pc_t.squeeze(0).cpu() for t, pc_t in traj.items()})
            else:
                # Normal sampling without trajectory
                generated_pc = model.sample(
                    z,
                    num_points=args.sample_num_points,
                    flexibility=args.flexibility,
                    text_emb=text_emb
                )  # (1, N, 3)

                generated_pc = generated_pc.squeeze(0).cpu()  # (N, 3)

                # Normalize
                if args.normalize != 'none':
                    generated_pc = normalize_point_cloud(generated_pc, mode=args.normalize)

            # Convert to numpy and append
            if torch.is_tensor(generated_pc):
                generated_pcs.append(generated_pc.numpy())
            else:
                generated_pcs.append(generated_pc)

    print(f"[SUCCESS] Generated {len(generated_pcs)} point clouds")

    # Save latent codes to file
    latent_codes_array = np.concatenate(latent_codes, axis=0)  # (num_samples, latent_dim)
    latent_codes_path = os.path.join(args.save_dir, 'latent_codes.npy')
    np.save(latent_codes_path, latent_codes_array)
    print(f"[INFO] Saved latent codes (z) to: {latent_codes_path}")
    print(f"      Shape: {latent_codes_array.shape}")

    # Save z statistics to text file
    z_stats_path = os.path.join(args.save_dir, 'latent_codes_stats.txt')
    with open(z_stats_path, 'w') as f:
        f.write(f"Latent Codes Statistics\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Number of samples: {len(latent_codes)}\n")
        f.write(f"Latent dimension: {latent_dim}\n")
        f.write(f"\n")

        for i, (z_val, caption) in enumerate(zip(latent_codes, selected_captions), 1):
            z_flat = z_val.flatten()
            f.write(f"Sample {i}: {caption}\n")
            f.write(f"  Mean: {z_flat.mean():.6f}, Std: {z_flat.std():.6f}\n")
            f.write(f"  Min: {z_flat.min():.6f}, Max: {z_flat.max():.6f}\n")
            f.write(f"  First 10 values: {z_flat[:10]}\n")
            f.write(f"\n")
    print(f"[INFO] Saved z statistics to: {z_stats_path}")

    # Visualize denoising process if requested
    if args.visualize_process and trajectories is not None:
        print(f"\n[INFO] Creating denoising process visualization...")
        process_path = os.path.join(args.save_dir, 'denoising_process.png')
        render_denoising_process(
            trajectories,
            selected_captions,
            save_path=process_path,
            num_steps=args.num_process_steps,
            figsize_per_cell=(args.grid_cell_width / 100, args.grid_cell_height / 100),
            point_size=args.point_size,
            color_scheme=args.color_scheme,
            angle=args.custom_angles[0] if args.custom_angles else 30
        )

    # Render point clouds to images
    print(f"\n[INFO] Rendering point clouds to 2D images using Open3D...")

    # Create subdirectory for renders
    render_dir = os.path.join(args.save_dir, 'renders')
    os.makedirs(render_dir, exist_ok=True)

    # Create grid visualization with rotation sequence
    if args.grid_layout:
        if args.use_3d:
            # Use matplotlib 3D plotting
            grid_path = os.path.join(args.save_dir, 'grid_visualization_3d.png')
            render_point_cloud_grid_3d(
                generated_pcs,
                selected_captions,
                save_path=grid_path,
                num_rotations=args.num_rotations,
                rotation_axis=args.rotation_axis,
                figsize_per_cell=(args.grid_cell_width / 100, args.grid_cell_height / 100),
                point_size=args.point_size,
                color_scheme=args.color_scheme,
                custom_angles=args.custom_angles
            )
        else:
            # Use Open3D 2D rendering
            grid_path = os.path.join(args.save_dir, 'grid_visualization.png')
            render_point_cloud_grid(
                generated_pcs,
                selected_captions,
                save_path=grid_path,
                num_rotations=args.num_rotations,
                rotation_axis=args.rotation_axis,
                img_width=args.grid_cell_width,
                img_height=args.grid_cell_height,
                point_size=args.point_size,
                bg_color=[1, 1, 1]
            )

        # Also create individual renders if requested
        if not args.grid_only:
            print(f"\n[INFO] Creating individual renders as well...")
            for i, (pc, caption) in enumerate(tqdm(list(zip(generated_pcs, selected_captions)), desc="Individual renders")):
                clean_caption = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in caption)
                clean_caption = clean_caption.replace(' ', '_')[:50]

                save_path = os.path.join(render_dir, f'sample_{i:04d}_{clean_caption}.png')
                render_point_cloud_to_image(
                    pc,
                    save_path=save_path,
                    window_name=caption,
                    width=args.render_width,
                    height=args.render_height,
                    point_size=args.point_size,
                    view_angle='default',
                    bg_color=[1, 1, 1]
                )
    else:
        # Original individual rendering
        for i, (pc, caption) in enumerate(tqdm(list(zip(generated_pcs, selected_captions)), desc="Rendering")):
            # Create clean caption for filename
            clean_caption = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in caption)
            clean_caption = clean_caption.replace(' ', '_')[:50]  # Limit length

            if args.render_multi_view:
                # Render from multiple angles
                angles = [
                    ('default', 'default'),
                    ('front', 'front'),
                    ('side', 'side'),
                    ('top', 'top'),
                    ('angle1', (30, 45)),
                    ('angle2', (30, 135)),
                ]

                for angle_name, angle_param in angles:
                    save_path = os.path.join(
                        render_dir,
                        f'sample_{i:04d}_{clean_caption}_{angle_name}.png'
                    )
                    render_point_cloud_to_image(
                        pc,
                        save_path=save_path,
                        window_name=f"{caption} ({angle_name})",
                        width=args.render_width,
                        height=args.render_height,
                        point_size=args.point_size,
                        view_angle=angle_param,
                        bg_color=[1, 1, 1]
                    )
            else:
                # Single default view
                save_path = os.path.join(
                    render_dir,
                    f'sample_{i:04d}_{clean_caption}.png'
                )
                render_point_cloud_to_image(
                    pc,
                    save_path=save_path,
                    window_name=caption,
                    width=args.render_width,
                    height=args.render_height,
                    point_size=args.point_size,
                    view_angle='default',
                    bg_color=[1, 1, 1]
                )

        print(f"[SUCCESS] All renders saved to: {render_dir}")

    # Interactive 3D viewer
    if args.interactive_viewer:
        visualize_interactive_3d(generated_pcs, selected_captions)

    # Save point clouds as numpy arrays
    if args.save_pointclouds:
        print(f"\n[INFO] Saving point clouds as numpy arrays...")
        pc_dir = os.path.join(args.save_dir, 'pointclouds')
        os.makedirs(pc_dir, exist_ok=True)

        for i, pc in enumerate(generated_pcs):
            np.save(
                os.path.join(pc_dir, f'sample_{i:04d}.npy'),
                pc
            )

        print(f"[INFO] Point clouds saved to: {pc_dir}")

    # Create summary report
    summary_file = os.path.join(args.save_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RANDOM CAPTION POINT CLOUD GENERATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Model type: {'FlowVAE' if 'latent_flow_depth' in ckpt['args'] else 'GaussianVAE'}\n")
        f.write(f"Iteration: {ckpt.get('iteration', 'unknown')}\n\n")
        f.write(f"Number of samples: {len(selected_captions)}\n")
        f.write(f"Points per cloud: {args.sample_num_points}\n")
        f.write(f"Normalization: {args.normalize}\n")
        f.write(f"Flexibility: {args.flexibility}\n\n")
        f.write("Generated Captions:\n")
        f.write("-"*80 + "\n")
        for i, caption in enumerate(selected_captions, 1):
            f.write(f"{i}. {caption}\n")
        f.write("="*80 + "\n")

    print(f"[INFO] Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print(f"[SUCCESS] All outputs saved to: {args.save_dir}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate and render point clouds from random custom captions using Open3D'
    )

    # Model checkpoint
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')

    # Captions
    caption_group = parser.add_mutually_exclusive_group(required=True)
    caption_group.add_argument('--captions_file', type=str,
                        help='Path to text file with captions (one per line)')
    caption_group.add_argument('--custom_captions', type=str, nargs='+',
                        default='./data/captions.tablechair.csv',
                        help='Custom captions from command line')

    # Sampling
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--sample_random', action='store_true',
                        help='Randomly sample from caption list (default: use first N)')
    parser.add_argument('--sample_num_points', type=int, default=2048,
                        help='Number of points to generate per cloud')
    parser.add_argument('--flexibility', type=float, default=0.0,
                        help='Sampling flexibility parameter')

    # Normalization
    parser.add_argument('--normalize', type=str, default='shape_unit',
                        choices=['shape_unit', 'shape_bbox', 'none'],
                        help='Point cloud normalization mode')

    # Rendering
    parser.add_argument('--render_width', type=int, default=800,
                        help='Width of rendered images (for individual renders)')
    parser.add_argument('--render_height', type=int, default=600,
                        help='Height of rendered images (for individual renders)')
    parser.add_argument('--point_size', type=float, default=5.0,
                        help='Point size in rendering')
    parser.add_argument('--render_multi_view', action='store_true',
                        help='Render from multiple viewing angles (individual mode only)')

    # Grid layout options
    parser.add_argument('--grid_layout', action='store_true',
                        help='Create grid visualization with rotation sequences')
    parser.add_argument('--grid_only', action='store_true',
                        help='Only create grid (skip individual renders)')
    parser.add_argument('--use_3d', action='store_true',
                        help='Use matplotlib 3D plotting instead of Open3D 2D rendering')
    parser.add_argument('--num_rotations', type=int, default=5,
                        help='Number of rotation angles in grid (columns, ignored if --custom_angles is set)')
    parser.add_argument('--custom_angles', type=float, nargs='+', default=None,
                        help='Specific rotation angles in degrees (e.g., --custom_angles 0 45 90 135 180)')
    parser.add_argument('--rotation_axis', type=str, default='y', choices=['y', 'x'],
                        help='Rotation axis: y=horizontal, x=vertical')
    parser.add_argument('--grid_cell_width', type=int, default=256,
                        help='Width of each grid cell (for 2D) or width in pixels/100 for 3D')
    parser.add_argument('--grid_cell_height', type=int, default=256,
                        help='Height of each grid cell (for 2D) or height in pixels/100 for 3D')

    # Interactive viewer
    parser.add_argument('--interactive_viewer', action='store_true',
                        help='Open interactive 3D viewer with Open3D (requires display)')

    # Color scheme
    parser.add_argument('--color_scheme', type=str, default='depth',
                        choices=['skyblue', 'gradient_blue', 'depth', 'height', 'gray'],
                        help='Color scheme for point clouds: skyblue (solid sky blue), gradient_blue (light to dark blue), depth (blue to red by depth), height (green by height), gray (uniform gray)')

    # Denoising process visualization
    parser.add_argument('--visualize_process', action='store_true',
                        help='Visualize the denoising/sampling process from noise to final point cloud')
    parser.add_argument('--num_process_steps', type=int, default=5,
                        help='Number of intermediate steps to visualize in denoising process (default: 5)')

    # Output
    parser.add_argument('--save_dir', type=str, default='./random_caption_visualizations',
                        help='Directory to save outputs')
    parser.add_argument('--save_pointclouds', action='store_true',
                        help='Save point clouds as numpy arrays')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (if None, uses current timestamp for random seed)')

    args = parser.parse_args()

    main(args)
