import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to Python path to import from thesis project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules from the thesis project
from nbjw_calib.utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    coords_to_dict,
    complete_keypoints
)
from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib_class import kp_to_line
from nbjw_calib.utils.utils_field import _draw_field
from pnlcalib.utils.utils_calib import FramebyFrameCalib
from tracklab.utils.download import download_file


class PitchDetector:
    def __init__(self, checkpoint_kp=None, checkpoint_l=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configuration for keypoint and line detection models
        self.cfg = {
            'MODEL': {
                'IMAGE_SIZE': [960, 540],
                'NUM_JOINTS': 58,
                'PRETRAIN': '',
                'EXTRA': {
                    'FINAL_CONV_KERNEL': 1,
                    'STAGE1': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 1,
                        'BLOCK': 'BOTTLENECK',
                        'NUM_BLOCKS': [4],
                        'NUM_CHANNELS': [64],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE2': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4],
                        'NUM_CHANNELS': [48, 96],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 4,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4, 4],
                        'NUM_CHANNELS': [48, 96, 192],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 3,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4, 4, 4],
                        'NUM_CHANNELS': [48, 96, 192, 384],
                        'FUSE_METHOD': 'SUM'
                    }
                }
            }
        }
        
        self.cfg_l = {
            'MODEL': {
                'IMAGE_SIZE': [960, 540],
                'NUM_JOINTS': 24,
                'PRETRAIN': '',
                'EXTRA': {
                    'FINAL_CONV_KERNEL': 1,
                    'STAGE1': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 1,
                        'BLOCK': 'BOTTLENECK',
                        'NUM_BLOCKS': [4],
                        'NUM_CHANNELS': [64],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE2': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4],
                        'NUM_CHANNELS': [48, 96],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 4,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4, 4],
                        'NUM_CHANNELS': [48, 96, 192],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 3,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4, 4, 4],
                        'NUM_CHANNELS': [48, 96, 192, 384],
                        'FUSE_METHOD': 'SUM'
                    }
                }
            }
        }
        
        # Setup model paths
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        if checkpoint_kp is None:
            checkpoint_kp = os.path.join(model_dir, 'SV_kp')
        if checkpoint_l is None:
            checkpoint_l = os.path.join(model_dir, 'SV_lines')
            
        # Download models if they don't exist
        if not os.path.isfile(checkpoint_kp):
            print(f"Downloading keypoint detection model to {checkpoint_kp}...")
            download_file("https://zenodo.org/records/12626395/files/SV_kp?download=1", checkpoint_kp)

        if not os.path.isfile(checkpoint_l):
            print(f"Downloading line detection model to {checkpoint_l}...")
            download_file("https://zenodo.org/records/12626395/files/SV_lines?download=1", checkpoint_l)
        
        # Load keypoint detection model
        loaded_state = torch.load(checkpoint_kp, map_location=self.device)
        self.model = get_cls_net(self.cfg)
        self.model.load_state_dict(loaded_state)
        self.model.to(self.device)
        self.model.eval()
        
        # Load line detection model
        loaded_state_l = torch.load(checkpoint_l, map_location=self.device)
        self.model_l = get_cls_net_l(self.cfg_l)
        self.model_l.load_state_dict(loaded_state_l)
        self.model_l.to(self.device)
        self.model_l.eval()
        
        # Transformations
        self.tfms_resize = T.Compose([
            T.Resize((540, 960)),
            T.ToTensor()
        ])
        
        # Initialize calibration module
        self.image_width = 960
        self.image_height = 540
        self.calibration = FramebyFrameCalib(self.image_width, self.image_height, denormalize=True)
    
    def preprocess_image(self, image):
        """Convert image to tensor and resize"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        image = self.tfms_resize(image)
        return image
    
    def detect_keypoints_and_lines(self, image_tensor):
        """Detect keypoints and lines in the image"""
        with torch.no_grad():
            heatmaps = self.model(image_tensor.to(self.device).unsqueeze(0))
            heatmaps_l = self.model_l(image_tensor.to(self.device).unsqueeze(0))
        
        # Extract keypoints and lines from heatmaps
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        
        # Convert coordinates to dictionaries
        kp_dict = coords_to_dict(kp_coords, threshold=0.1449)
        lines_dict = coords_to_dict(line_coords, threshold=0.2983)
        
        # Complete keypoints based on detected lines
        image_width = image_tensor.size()[-1]
        image_height = image_tensor.size()[-2]
        final_dict = complete_keypoints(kp_dict, lines_dict, w=image_width, h=image_height, normalize=True)

        output_pred = []
        for result in final_dict:
            output_pred.append({"keypoints": result, "lines": kp_to_line(result)})

        return output_pred
    
    def get_homography(self, keypoints, lines):
        """Calculate homography matrix from detected keypoints"""
        self.calibration.update(keypoints, lines)
        homography = self.calibration.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        return homography
    
    def visualize_pitch(self, image, keypoints, lines, homography=None, save_path=None):
        """Visualize detected pitch on the image"""
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        
        # Create a copy of the image for visualization
        viz_image = image.copy()
        
        # Draw keypoints
        for kp_id, kp in keypoints.items():
            x, y = int(kp['x'] * self.image_width), int(kp['y'] * self.image_height)
            cv2.circle(viz_image, (x, y), 3, (0, 255, 0), -1)
        
        # Draw lines
        for line_name, line_points in lines.items():
            if len(line_points) >= 2:
                pt1 = (int(line_points[0]['x'] * self.image_width), int(line_points[0]['y'] * self.image_height))
                pt2 = (int(line_points[1]['x'] * self.image_width), int(line_points[1]['y'] * self.image_height))
                cv2.line(viz_image, pt1, pt2, (255, 0, 0), 2)
        
        # Draw field overlay if homography is available
        if homography is not None:
            _draw_field(viz_image, homography)
        
        # Save or display the result
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
        
        return viz_image
    
    def process_image(self, image, visualize=True, save_path=None):
        """Process a single image and return detected pitch elements"""
        # Preprocess the image
        image_tensor = self.preprocess_image(image)
        
        # Detect keypoints and lines
        output_pred = self.detect_keypoints_and_lines(image_tensor)


        # Initialize list to store homography matrices
        homography_matrices = []
        
        # Get homography for each prediction
        for item in output_pred:
            keypoints = item["keypoints"]
            lines = item["lines"]
            
            homography = self.get_homography(keypoints, lines)
            if homography is not None:
                homography_matrices.append(homography)
        
        # Calculate average homography if we have valid matrices
        if homography_matrices:
            homography = np.mean(homography_matrices, axis=0)
        else:
            homography = None
        
        # Visualize results if requested
        if visualize:
            viz_image = self.visualize_pitch(image, keypoints, lines, homography, save_path)
            return keypoints, lines, homography, viz_image
        
        return keypoints, lines, homography
    
    def process_video(self, video_path, output_path=None, start_frame=0, max_frames=None, fps=None):
        """Process a video file and detect pitch in each frame"""
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set output fps
        if fps is None:
            fps = original_fps
        
        # Set up output video writer if needed
        out = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process only the specified range of frames
        if max_frames is None:
            max_frames = total_frames - start_frame
        
        # Skip to the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        try:
            # Process frames
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                _, _, homography, viz_frame = self.process_image(frame_rgb, visualize=True)
                
                # Convert back to BGR for video writing
                viz_frame_bgr = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)
                
                # Resize back to original resolution if needed
                if viz_frame_bgr.shape[0] != height or viz_frame_bgr.shape[1] != width:
                    viz_frame_bgr = cv2.resize(viz_frame_bgr, (width, height))
                
                # Write the frame
                if out:
                    out.write(viz_frame_bgr)
                
                # Print progress
                if i % 10 == 0:
                    print(f"Processed {i+1}/{max_frames} frames")
        
        finally:
            # Release resources
            cap.release()
            if out:
                out.release()
        
        print(f"Video processing complete. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect soccer pitch lines in images or videos")
    parser.add_argument("input_path", help="Path to input image or video file")
    parser.add_argument("--output_path", help="Path to save output visualization")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame for video processing")
    parser.add_argument("--max_frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--fps", type=float, help="Output video fps (defaults to input video fps)")
    
    args = parser.parse_args()
    
    # Initialize the pitch detector
    detector = PitchDetector()
    
    # Check if input is an image or video
    input_path = args.input_path
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        # Process image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Failed to load image file '{input_path}'. Please check that the file exists and is a valid image.")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        output_path = args.output_path
        if output_path is None:
            output_path = f"{os.path.splitext(input_path)[0]}_pitch_detected.jpg"
        
        _, _, _, _ = detector.process_image(image_rgb, save_path=output_path)
        print(f"Image processing complete. Output saved to: {output_path}")
    
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video
        if not os.path.exists(input_path):
            print(f"Error: Video file '{input_path}' does not exist. Please check the file path.")
            return
            
        output_path = args.output_path
        if output_path is None:
            output_path = f"{os.path.splitext(input_path)[0]}_pitch_detected.mp4"
        
        detector.process_video(
            input_path, 
            output_path=output_path,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            fps=args.fps
        )
    
    else:
        print(f"Unsupported file format: {input_path}")


if __name__ == "__main__":
    main() 