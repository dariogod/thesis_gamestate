#!/usr/bin/env python

import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from yacs.config import CfgNode as CN
from omegaconf import OmegaConf
import numpy
import shutil

# Import the real PRTReId implementation
from thesis_sn_gamestate.sn_gamestate.reid.prtreid_api import PRTReId
from tracklab.utils.collate import Unbatchable
from tracklab.datastruct import TrackingDataset, TrackingSet

# Fix for PyTorch weight loading (needed since PyTorch 2.6)
try:
    # Add numpy scalar to safe globals for PyTorch weight loading
    import torch.serialization
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
    # Alternatively, could patch the torch.load function in prtreid
    original_torch_load = torch.load
    def patched_torch_load(f, *args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)
    torch.load = patched_torch_load
    print("Applied PyTorch weight loading patches")
except Exception as e:
    print(f"Warning: Failed to patch PyTorch loading: {e}")

# Create a simple BBox class to mimic detection.bbox from tracklab
class BBox:
    def __init__(self, ltwh):
        self.ltwh_values = ltwh
    
    def ltwh(self, image_shape=None, rounded=False):
        return self.ltwh_values
    
    def ltrb(self, image_shape=None, rounded=False):
        l, t, w, h = self.ltwh_values
        return l, t, l+w, t+h

def load_annotations(json_path):
    """Load annotations from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_bboxes(annotations, image_id):
    """Extract bounding boxes from annotations for a specific image."""
    bboxes = []
    
    # Filter annotations for the specific image_id
    for ann in annotations["annotations"]:
        if ann['image_id'] != image_id:
            continue
            
        # Skip the ball (category_id 4)
        if ann['category_id'] == 4:
            continue

        if "bbox_image" not in ann:
            continue

        # Convert bbox values to numpy array
        bbox_ltwh = np.array([
            ann['bbox_image']['x'], 
            ann['bbox_image']['y'], 
            ann['bbox_image']['w'], 
            ann['bbox_image']['h']
        ], dtype=np.float32)
        
        bbox_info = {
            'id': ann['id'],
            'track_id': ann['track_id'],
            'bbox': BBox([
                ann['bbox_image']['x'], 
                ann['bbox_image']['y'], 
                ann['bbox_image']['w'], 
                ann['bbox_image']['h']
            ]),
            'bbox_ltwh': bbox_ltwh,  # Use numpy array instead of list
            'role': ann['attributes']['role'],
            'jersey': ann['attributes']['jersey'],
            'team': ann['attributes']['team'],
            'category_id': ann['category_id']
        }
        bboxes.append(bbox_info)
    return bboxes

def setup_prtreid_config():
    """Set up the configuration for PRTReId."""
    # Directory where model weights will be stored
    model_dir = os.path.expanduser("~/models/reid")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a full config for PRTReId based on actual usage in the codebase
    cfg = CN()
    cfg.project = CN()
    cfg.project.name = "TrackLab"
    cfg.project.experiment_name = ""
    cfg.project.notes = ""
    cfg.project.tags = []
    cfg.project.logger = CN()
    cfg.project.logger.use_tensorboard = False
    cfg.project.logger.use_wandb = False
    
    cfg.data = CN()
    cfg.data.root = f"{model_dir}/reid"
    cfg.data.type = "image"
    cfg.data.sources = ["SoccerNet"]
    cfg.data.targets = ["SoccerNet"]
    cfg.data.height = 256
    cfg.data.width = 128
    cfg.data.combineall = False
    cfg.data.transforms = ["rc", "re"]
    cfg.data.save_dir = ""
    cfg.data.workers = 4
    
    cfg.sampler = CN()
    cfg.sampler.train_sampler = "PrtreidSampler"
    cfg.sampler.train_sampler_t = "PrtreidSampler"
    cfg.sampler.num_instances = 4
    
    cfg.model = CN()
    cfg.model.name = "bpbreid"
    cfg.model.pretrained = True
    cfg.model.save_model_flag = True
    cfg.model.load_config = True
    cfg.model.load_weights = f"{model_dir}/prtreid-soccernet-baseline.pth.tar"
    
    cfg.model.bpbreid = CN()
    cfg.model.bpbreid.pooling = "gwap"
    cfg.model.bpbreid.normalization = "identity"
    cfg.model.bpbreid.mask_filtering_training = False
    cfg.model.bpbreid.mask_filtering_testing = False
    cfg.model.bpbreid.training_binary_visibility_score = True
    cfg.model.bpbreid.testing_binary_visibility_score = True
    cfg.model.bpbreid.last_stride = 1
    cfg.model.bpbreid.learnable_attention_enabled = False
    cfg.model.bpbreid.dim_reduce = "after_pooling"
    cfg.model.bpbreid.dim_reduce_output = 256
    cfg.model.bpbreid.backbone = "hrnet32"
    cfg.model.bpbreid.test_embeddings = ["globl"]
    cfg.model.bpbreid.test_use_target_segmentation = "none"
    cfg.model.bpbreid.shared_parts_id_classifier = False
    cfg.model.bpbreid.hrnet_pretrained_path = model_dir
    
    cfg.model.bpbreid.masks = CN()
    cfg.model.bpbreid.masks.type = "disk"
    cfg.model.bpbreid.masks.dir = "pose_on_img_crops"
    cfg.model.bpbreid.masks.preprocess = "id"
    
    cfg.loss = CN()
    cfg.loss.name = 'part_based'
    
    cfg.loss.part_based = CN()
    cfg.loss.part_based.name = 'part_averaged_triplet_loss'
    cfg.loss.part_based.ppl = 'cl'
    
    cfg.loss.part_based.weights = CN()
    cfg.loss.part_based.weights.globl = CN()
    cfg.loss.part_based.weights.globl.id = 1.0
    cfg.loss.part_based.weights.globl.tr = 1.0
    
    cfg.loss.part_based.weights.foreg = CN()
    cfg.loss.part_based.weights.foreg.id = 0.0
    cfg.loss.part_based.weights.foreg.tr = 0.0
    
    cfg.loss.part_based.weights.conct = CN()
    cfg.loss.part_based.weights.conct.id = 0.0
    cfg.loss.part_based.weights.conct.tr = 0.0
    
    cfg.loss.part_based.weights.parts = CN()
    cfg.loss.part_based.weights.parts.id = 0.0
    cfg.loss.part_based.weights.parts.tr = 0.0
    
    cfg.loss.part_based.weights.pixls = CN()
    cfg.loss.part_based.weights.pixls.ce = 0.0
    
    cfg.train = CN()
    cfg.train.batch_size = 32
    cfg.train.max_epoch = 20
    
    # Create dataset config
    dataset_cfg = CN()
    dataset_cfg.name = "SoccerNet"
    dataset_cfg.nickname = "sn"
    dataset_cfg.fig_size = [384, 128]
    dataset_cfg.mask_size = [64, 32]
    dataset_cfg.max_crop_size = [256, 128]
    dataset_cfg.masks_mode = "pose_on_img_crops"
    dataset_cfg.enable_human_parsing_labels = False
    dataset_cfg.eval_metric = "mot_intra_video"
    dataset_cfg.columns = ["role", "team", "jersey_number"]
    dataset_cfg.multi_video_queries_only = False
    
    dataset_cfg.train = CN()
    dataset_cfg.train.set_name = "train"
    dataset_cfg.train.min_vis = 0.3
    dataset_cfg.train.min_h = 30
    dataset_cfg.train.min_w = 30
    dataset_cfg.train.min_samples_per_id = 4
    dataset_cfg.train.max_samples_per_id = 15
    dataset_cfg.train.max_total_ids = -1
    
    dataset_cfg.test = CN()
    dataset_cfg.test.set_name = "valid"
    dataset_cfg.test.min_vis = 0.0
    dataset_cfg.test.min_h = 0
    dataset_cfg.test.min_w = 0
    dataset_cfg.test.min_samples_per_id = 4
    dataset_cfg.test.max_samples_per_id = 10
    dataset_cfg.test.max_total_ids = -1
    dataset_cfg.test.ratio_query_per_id = 0.2
    
    return cfg, dataset_cfg, model_dir

def process_image(model, image_path, bboxes, output_dir):
    """Process a single image with the PRTReId model."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Convert BGR to RGB (prtreid expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process each detection
    results = []
    
    for i, bbox_info in enumerate(bboxes):
        # Skip the ball (category_id 4)
        if bbox_info['category_id'] == 4:
            continue
            
        # Create detection Series
        detection = pd.Series({
            'bbox': bbox_info['bbox'],
            'bbox_ltwh': bbox_info['bbox_ltwh'],
            'id': i+1,
            'video_id': 1,
            'image_id': 1,
            'person_id': bbox_info['track_id'],
            'visibility': 1.0
        })
        
        # Create metadata
        metadata = pd.Series({
            'id': 1,
            'video_id': 1,
            'frame': 0,
            'file_path': image_path
        })
        
        try:            
            # Preprocess
            batch = model.preprocess(image, detection, metadata)
            
            # Create DataFrames for process function
            detections_df = pd.DataFrame([detection], index=[i+1])
            metadata_df = pd.DataFrame([metadata], index=[1])
            
            # Run the model
            reid_df = model.process(batch, detections_df, metadata_df)
            
            # Add annotation attributes for comparison
            result = {
                'id': bbox_info['id'],
                'track_id': bbox_info['track_id'],
                'bbox': [
                    bbox_info['bbox'].ltwh()[0],
                    bbox_info['bbox'].ltwh()[1],
                    bbox_info['bbox'].ltwh()[2],
                    bbox_info['bbox'].ltwh()[3]
                ],
                'true_role': bbox_info['role'],
                'true_jersey': bbox_info['jersey'],
                'true_team': bbox_info['team'],
                'predicted_role': reid_df['role_detection'][i+1],
                'role_confidence': reid_df['role_confidence'][i+1],
                'embedding_shape': reid_df['embeddings'][i+1].shape
            }
            results.append(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing detection {i+1}: {e}")
    
    if not results:
        print("No valid results for this image")
        return None
    
    # Create a visualization of the image with boxes
    vis_image = cv2.imread(image_path)
    if vis_image is None:
        print(f"Warning: Could not load image for visualization: {image_path}")
        return results
    
    # Draw bounding boxes and labels
    for result in results:
        l, t, w, h = result['bbox']
        
        # Different colors for different teams
        if result['true_team'] == 'left':
            color = (255, 255, 0)  # Cyan
        elif result['true_team'] == 'right':
            color = (0, 0, 255)    # Red
        else:
            color = (255, 255, 0)  # Yellow (referee)
            
        # Draw bounding box
        cv2.rectangle(vis_image, (int(l), int(t)), (int(l+w), int(t+h)), color, 2)
        
        # Draw labels with true and predicted roles
        true_label = f"T: {result['true_role']}"
        if result['true_jersey']:
            true_label += f" #{result['true_jersey']}"
            
        pred_label = f"P: {result['predicted_role']} ({result['role_confidence']:.2f})"
        
        cv2.putText(vis_image, true_label, (int(l), int(t-25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(vis_image, pred_label, (int(l), int(t-5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization
    cv2.imwrite(f"{output_dir}/visualized_detections.jpg", vis_image)
    print(f"\nSaved visualization to {output_dir}/visualized_detections.jpg")
    
    return results

def main():
    SNGS_ID = "SNGS-195"

    input_dir = f"role_assignment/input/{SNGS_ID}"
    base_output_dir = f"role_assignment/output/{SNGS_ID}"
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)

    # Load annotation file
    annotation_file = f"{input_dir}/Labels-GameState.json"
    
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file {annotation_file} not found")
        return
        
    with open(annotation_file, "r") as f:
        labels = json.load(f)
    
    # Get list of images
    image_dir = os.path.join(input_dir, "img1")
    img_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    img_files = sorted(img_files)
    img_files = [img_file for img_file in img_files if int(img_file.split('.')[0]) % 10 == 0]

    
    print(f"Found {len(img_files)} images to process")
    
    # Build a map of filename to image_id
    filename_to_id = {}
    for img_info in labels["images"]:
        filename_to_id[img_info["file_name"]] = img_info["image_id"]
    
    # Set up PRTReId
    print("Setting up PRTReId model...")
    cfg, dataset_cfg, model_dir = setup_prtreid_config()
    
    # Create a minimal tracking dataset
    empty_df = pd.DataFrame()
    tracking_set = TrackingSet(empty_df, empty_df, empty_df)
    tracking_sets = {"train": tracking_set}
    
    class MinimalTrackingDataset(TrackingDataset):
        def __init__(self, dataset_path, sets):
            self.dataset_path = Path(dataset_path)
            self.sets = sets
            self.name = "SoccerNet"
            self.nickname = "sn"
    
    tracking_dataset = MinimalTrackingDataset("./", tracking_sets)
    
    # Check if model weights exist, otherwise inform user
    weights_path = Path(cfg.model.load_weights)
    hrnet_path = Path(cfg.model.bpbreid.hrnet_pretrained_path) / "hrnetv2_w32_imagenet_pretrained.pth"
    
    if not weights_path.exists():
        print(f"Warning: Model weights not found at {weights_path}")
        print("The model will attempt to download weights or may fail if downloads are not available.")
    
    if not hrnet_path.exists():
        print(f"Warning: HRNet weights not found at {hrnet_path}")
        print("The model will attempt to download weights or may fail if downloads are not available.")

    # Clean up existing output directory
    save_path = "reid"
    job_id = "0"
    temp_output_dir = os.path.join(save_path, job_id)
    if os.path.exists(temp_output_dir):
        print(f"Removing existing temporary output directory: {temp_output_dir}")
        shutil.rmtree(temp_output_dir, ignore_errors=True)

    # Initialize the PRTReId model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Use string device instead of torch.device object for compatibility
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Convert CN to OmegaConf for PRTReId
        cfg_dict = cfg.dump()
        omega_cfg = OmegaConf.create(cfg_dict)
        
        dataset_cfg_dict = dataset_cfg.dump()
        omega_dataset_cfg = OmegaConf.create(dataset_cfg_dict)
        
        # Additional fix for prtreid torchtools.py load_checkpoint
        try:
            import prtreid.utils.torchtools
            original_load_checkpoint = prtreid.utils.torchtools.load_checkpoint
            
            def patched_load_checkpoint(fpath, map_location=None):
                print(f"Using patched load_checkpoint for {fpath}")
                if map_location is None:
                    map_location = 'cpu'  # Always use CPU to ensure compatibility
                return torch.load(fpath, map_location=map_location, weights_only=False)
                
            prtreid.utils.torchtools.load_checkpoint = patched_load_checkpoint
            print("Successfully patched prtreid.utils.torchtools.load_checkpoint")
        except Exception as e:
            print(f"Warning: Could not patch prtreid load_checkpoint: {e}")
        
        model = PRTReId(
            cfg=omega_cfg,
            tracking_dataset=tracking_dataset,
            dataset=omega_dataset_cfg,
            device=device_str,
            save_path=save_path,
            job_id=job_id,
            use_keypoints_visibility_scores_for_reid=False,
            training_enabled=False,
            batch_size=1
        )
        print("PRTReId model initialized successfully")
        
    except Exception as e:
        print(f"Error initializing PRTReId: {e}")
        raise e
    
    # Process each image
    all_results = {}
        
    for i, img_file in enumerate(img_files):
        print(f"\nProcessing image {i+1}/{len(img_files)}: {img_file}")
        
        # Create output directory for this image
        img_name_without_ext = os.path.splitext(img_file)[0]
        img_output_dir = os.path.join(base_output_dir, img_name_without_ext)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Get image_id from filename
        if img_file in filename_to_id:
            image_id = filename_to_id[img_file]
        else:
            print(f"Warning: Could not find image_id for {img_file}, skipping")
            continue
        
        # Get bounding boxes for this image
        bboxes = extract_bboxes(labels, image_id)
        print(f"Found {len(bboxes)} detections (excluding ball) for image {img_file}")
        
        if not bboxes:
            print(f"No valid detections found for image {img_file}, skipping")
            continue
        
        # Process the image
        img_path = os.path.join(image_dir, img_file)
        results = process_image(model, img_path, bboxes, img_output_dir)
        
        if results:
            # Save results for this image
            with open(f"{img_output_dir}/reid_results.json", "w") as f:
                # Convert numpy shapes to lists for JSON serialization
                for r in results:
                    if isinstance(r['embedding_shape'], tuple):
                        r['embedding_shape'] = list(r['embedding_shape'])
                json.dump(results, f, indent=2)
            
            print(f"Saved detailed results to {img_output_dir}/reid_results.json")
            all_results[img_file] = results
            print(f"Saved results for {img_file}")
        
        else:
            print(f"No results for {img_file}")

    
    # Save all results in a single file
    with open(f"{base_output_dir}/all_reid_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nProcessed {len(all_results)} images successfully")
    print(f"All results saved to {base_output_dir}/all_reid_results.json")

if __name__ == "__main__":
    main() 