# visualize_model_predictions.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from tqdm import tqdm

# Add project root directory to path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import related modules
from src.models.unet import UNet
from src.models.attention_unet import get_attention_unet_model
from dataloader import LiTSDataset, get_transforms


def load_trained_models(models_dir, device, model_types=None):
    """
    Load trained models

    Parameters:
        models_dir: Model directory
        device: Computing device
        model_types: List of model types to load, if None, load all models

    Returns:
        Model dictionary {model_name: model}
    """
    models_dir = Path(models_dir)
    models = {}

    # Traverse model directory
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_path = model_dir / 'best_model.pth'
        config_path = model_dir / 'config.json'

        if model_path.exists() and config_path.exists():
            try:
                # Read configuration
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Extract model type and parameters
                model_type = config.get('model_type', 'standard')

                # Skip if model types are specified and current model not in list
                if model_types is not None and model_type not in model_types:
                    continue

                # Simplify model name
                model_name = model_dir.name.split('_')[0]
                if len(model_dir.name.split('_')) > 1:
                    attention_type = model_dir.name.split('_')[1]
                    if attention_type not in ['20', '202']:  # Exclude timestamp part
                        model_name += f"_{attention_type}"

                # Create model
                n_channels = config.get('n_channels', 3)
                n_classes = config.get('n_classes', 3)
                init_features = config.get('init_features', 64)

                # Load model parameters
                if model_type == 'standard':
                    model = UNet(
                        n_channels=n_channels,
                        n_classes=n_classes,
                        init_features=init_features
                    ).to(device)
                else:
                    model = get_attention_unet_model(
                        model_type=model_type,
                        n_channels=n_channels,
                        n_classes=n_classes,
                        init_features=init_features,
                        attention_type=config.get('attention_type', 'cbam')
                    ).to(device)

                # Load model weights
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                # Add to model dictionary
                models[model_name] = model
                print(f"Loaded model: {model_name}")

            except Exception as e:
                print(f"Error loading model {model_dir.name}: {e}")

    return models


def find_small_tumor_samples(val_loader, num_samples=5):
    """
    Find samples containing small tumors in validation set

    Parameters:
        val_loader: Validation data loader
        num_samples: Number of samples to find

    Returns:
        List of samples containing small tumors
    """
    small_tumor_samples = []
    sample_count = 0

    print("Looking for samples containing small tumors...")

    for batch in tqdm(val_loader):
        # Check if batch contains weight_map (for identifying small tumors)
        if 'weight_map' not in batch:
            continue

        # Find samples with high weight values (corresponding to small tumors)
        for i in range(len(batch['image'])):
            # Check if it contains tumor
            if not torch.any(batch['mask'][i] == 2):
                continue

            # Check if it contains high weight areas (small tumors)
            weight_map = batch['weight_map'][i]
            if torch.any(weight_map > 0.8):
                # Extract sample
                sample = {
                    'image': batch['image'][i],
                    'mask': batch['mask'][i],
                    'weight_map': weight_map,
                    'file_name': batch['file_name'][i] if 'file_name' in batch else f"sample_{sample_count}"
                }
                small_tumor_samples.append(sample)
                sample_count += 1

                if sample_count >= num_samples:
                    return small_tumor_samples

    print(f"Found {len(small_tumor_samples)} samples containing small tumors")
    return small_tumor_samples


def predict_samples(models, samples, device):
    """
    Use all models to predict samples

    Parameters:
        models: Model dictionary {model_name: model}
        samples: Sample list
        device: Computing device

    Returns:
        Prediction result dictionary {model_name: {sample_index: prediction_result}}
    """
    predictions = {model_name: {} for model_name in models}

    for i, sample in enumerate(samples):
        image = sample['image'].unsqueeze(0).to(device)

        for model_name, model in models.items():
            with torch.no_grad():
                output = model(image)
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1)[0].cpu().numpy()
                predictions[model_name][i] = pred

    return predictions


def visualize_predictions(samples, predictions, models_to_show=None, save_dir=None):
    """
    Visualize model prediction results

    Parameters:
        samples: Sample list
        predictions: Prediction result dictionary
        models_to_show: List of models to display, if None, display all models
        save_dir: Save directory
    """
    if models_to_show is None:
        models_to_show = list(predictions.keys())

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        # Prepare image and labels
        image = sample['image'].numpy().transpose(1, 2, 0)  # CHW -> HWC
        mask = sample['mask'].numpy()
        weight_map = sample['weight_map'].numpy()
        file_name = sample['file_name']

        # Calculate image layout
        num_models = len(models_to_show)
        total_cols = 3  # Image, Label, Weight map
        total_rows = 1 + (num_models + 1) // 2  # Original row + model row(s)

        fig = plt.figure(figsize=(total_cols * 5, total_rows * 4))

        # Display original image
        plt.subplot(total_rows, total_cols, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        # Display ground truth label
        plt.subplot(total_rows, total_cols, 2)
        plt.imshow(mask, cmap='viridis', vmin=0, vmax=2)
        plt.title('Ground Truth (0=Background, 1=Liver, 2=Tumor)')
        plt.axis('off')

        # Display weight map
        plt.subplot(total_rows, total_cols, 3)
        if len(weight_map.shape) == 3 and weight_map.shape[0] == 1:
            weight_map = weight_map[0]  # If 3D array, take first channel
        plt.imshow(weight_map, cmap='hot')
        plt.title('Small Tumor Weight Map')
        plt.axis('off')

        # Display each model's predictions
        for j, model_name in enumerate(models_to_show):
            if i in predictions[model_name]:
                pred = predictions[model_name][i]
                plt.subplot(total_rows, total_cols, 4 + j)
                plt.imshow(pred, cmap='viridis', vmin=0, vmax=2)
                plt.title(f'{model_name} Prediction')
                plt.axis('off')

        plt.tight_layout()

        # Save image
        if save_dir:
            plt.savefig(save_dir / f"sample_{i}_{file_name}.png")
            plt.close()
        else:
            plt.show()


def visualize_small_tumor_regions(samples, predictions, models_to_show=None, save_dir=None):
    """
    Visualize model prediction results for small tumor regions

    Parameters:
        samples: Sample list
        predictions: Prediction result dictionary
        models_to_show: List of models to display, if None, display all models
        save_dir: Save directory
    """
    if models_to_show is None:
        models_to_show = list(predictions.keys())

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        # Prepare image and labels
        image = sample['image'].numpy().transpose(1, 2, 0)  # CHW -> HWC
        mask = sample['mask'].numpy()

        # Process weight map to ensure dimensions are correct
        weight_map = sample['weight_map'].numpy()
        if len(weight_map.shape) == 3 and weight_map.shape[0] == 1:
            weight_map = weight_map[0]  # If 3D array, take first channel

        file_name = sample['file_name']

        # Find tumor area - directly use tumor pixel positions in mask
        tumor_pos = np.where(mask == 2)

        if len(tumor_pos[0]) == 0:
            print(f"No tumor in sample {file_name}, skipping")
            continue

        # Calculate center of all tumor pixels
        center_y, center_x = int(np.mean(tumor_pos[0])), int(np.mean(tumor_pos[1]))

        # Find bounding box of tumor region, slightly enlarged for better visualization
        tumor_min_y, tumor_max_y = np.min(tumor_pos[0]), np.max(tumor_pos[0])
        tumor_min_x, tumor_max_x = np.min(tumor_pos[1]), np.max(tumor_pos[1])

        # Calculate center and size of bounding box
        tumor_center_y = (tumor_min_y + tumor_max_y) // 2
        tumor_center_x = (tumor_min_x + tumor_max_x) // 2

        # Set zoom region size, ensure it at least contains the tumor plus some margin
        height = max(64, tumor_max_y - tumor_min_y + 40)
        width = max(64, tumor_max_x - tumor_min_x + 40)
        region_size = max(height, width)

        # Calculate region boundaries, ensure they are within the image
        y_start = max(0, tumor_center_y - region_size // 2)
        y_end = min(mask.shape[0], tumor_center_y + region_size // 2)
        x_start = max(0, tumor_center_x - region_size // 2)
        x_end = min(mask.shape[1], tumor_center_x + region_size // 2)

        # Further adjust if region is too small
        if y_end - y_start < 64:
            if y_start == 0:
                y_end = min(mask.shape[0], y_end + (64 - (y_end - y_start)))
            else:
                y_start = max(0, y_start - (64 - (y_end - y_start)))

        if x_end - x_start < 64:
            if x_start == 0:
                x_end = min(mask.shape[1], x_end + (64 - (x_end - x_start)))
            else:
                x_start = max(0, x_start - (64 - (x_end - x_start)))

        # Calculate image layout
        num_models = len(models_to_show)
        total_cols = min(4, num_models + 2)  # Image, Label, Model predictions
        total_rows = 1 + (num_models + 1) // (total_cols - 1)  # At least one row

        fig = plt.figure(figsize=(total_cols * 5, total_rows * 4))

        # Display original image region
        plt.subplot(total_rows, total_cols, 1)
        plt.imshow(image[y_start:y_end, x_start:x_end])
        plt.title(f'Original Image Region - Sample {file_name}')
        plt.axis('off')

        # Display ground truth label region
        plt.subplot(total_rows, total_cols, 2)
        # Use more vivid color mapping to make tumors more obvious
        # Create a custom color map: purple=background, teal=liver, yellow=tumor
        cmap = plt.cm.colors.ListedColormap(['purple', 'teal', 'yellow'])
        plt.imshow(mask[y_start:y_end, x_start:x_end], cmap=cmap, vmin=0, vmax=2)
        plt.title('Ground Truth Tumor Region')
        plt.axis('off')

        # Draw tumor region contour to highlight
        tumor_mask_region = mask[y_start:y_end, x_start:x_end] == 2
        if np.any(tumor_mask_region):
            from skimage import measure
            contours = measure.find_contours(tumor_mask_region.astype(np.float32), 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)

        # Display predictions from each model
        for j, model_name in enumerate(models_to_show[:min(len(models_to_show), total_cols * total_rows - 2)]):
            if i in predictions[model_name]:
                pred = predictions[model_name][i]
                plt.subplot(total_rows, total_cols, 3 + j)
                pred_region = pred[y_start:y_end, x_start:x_end]
                plt.imshow(pred_region, cmap=cmap, vmin=0, vmax=2)

                # Calculate tumor detection accuracy in this region
                truth_tumor = mask[y_start:y_end, x_start:x_end] == 2
                pred_tumor = pred_region == 2
                if np.any(truth_tumor):
                    # Calculate recall and precision for this region
                    region_recall = np.sum(pred_tumor & truth_tumor) / np.sum(truth_tumor)
                    region_precision = np.sum(pred_tumor & truth_tumor) / np.sum(pred_tumor) if np.sum(
                        pred_tumor) > 0 else 0
                    region_f1 = 2 * region_recall * region_precision / (region_recall + region_precision) if (
                                                                                                                         region_recall + region_precision) > 0 else 0

                    plt.title(f'{model_name}\nR={region_recall:.2f}, P={region_precision:.2f}, F1={region_f1:.2f}')
                else:
                    plt.title(f'{model_name}')

                # Draw predicted tumor contours
                pred_tumor_contours = measure.find_contours(pred_tumor.astype(np.float32), 0.5)
                for contour in pred_tumor_contours:
                    plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1)

                # Draw ground truth tumor contours for comparison
                truth_tumor_contours = measure.find_contours(truth_tumor.astype(np.float32), 0.5)
                for contour in truth_tumor_contours:
                    plt.plot(contour[:, 1], contour[:, 0], 'g--', linewidth=1)

                plt.axis('off')

        plt.suptitle(f"Small Tumor Region Prediction Comparison - Sample {file_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save image
        if save_dir:
            plt.savefig(save_dir / f"small_tumor_region_{i}_{file_name}.png")
            plt.close()
        else:
            plt.show()


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained models
    models = load_trained_models(args.models_dir, device, args.model_types)

    if not models:
        print("No models found")
        return

    # Prepare validation data loader
    val_transform = get_transforms('val', img_size=tuple(map(int, args.img_size.split(','))))

    val_dataset = LiTSDataset(
        data_dir=args.data_dir,
        slice_list_file=Path(args.data_dir) / "splits" / "val_slices.txt",
        transform=val_transform,
        phase="val",
        small_tumor_focus=True,
        return_weight_map=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Validation set size: {len(val_dataset)} slices")

    # Find samples containing small tumors
    small_tumor_samples = find_small_tumor_samples(val_loader, args.num_samples)

    if not small_tumor_samples:
        print("No samples containing small tumors found")
        return

    # Use all models to predict samples
    predictions = predict_samples(models, small_tumor_samples, device)

    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize prediction results
    visualize_predictions(
        small_tumor_samples,
        predictions,
        models_to_show=args.model_types,
        save_dir=output_dir / "full_predictions"
    )

    # Visualize small tumor regions
    visualize_small_tumor_regions(
        small_tumor_samples,
        predictions,
        models_to_show=args.model_types,
        save_dir=output_dir / "small_tumor_regions"
    )

    print(f"Visualization results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model prediction results')
    parser.add_argument('--models_dir', type=str, default='results/models/attention',
                        help='Model directory')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                        help='Preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default='results/visualizations',
                        help='Output directory')
    parser.add_argument('--img_size', type=str, default='256,256',
                        help='Image size, format "height,width"')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--model_types', nargs='+', default=None,
                        help='Model types to include, if not specified, include all models')

    args = parser.parse_args()
    main(args)