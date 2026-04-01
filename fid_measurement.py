import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


class FetalPlaneDataset(Dataset):
    """Fetal Plane dataset."""
    def __init__(self,
                root_dir,
                csv_file,
                transform=None,
                split='train',
                plane_filter=None
                ):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string, optional): Which part of the dataset are we using (train or test)
            plane_filter (int, optional): Filter to specific plane class
        """
        self.transform = transform
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file, sep=';')
        self.split = split
        
        # Which dataset split are using? Training or test?
        if self.split == "train":
            self.df = self.df[self.df['Train'] == 1]
        elif self.split == "test":
            self.df = self.df[self.df['Train'] == 0]
        
        # Define classes and give each int labels
        self.plane_classes = {
            'Other' : 0, 
            'Maternal cervix' : 1, 
            'Fetal abdomen' : 2, 
            'Fetal brain' : 3, 
            'Fetal femur' : 4, 
            'Fetal thorax' : 5,
        }
        
        # Filter by plane class if specified
        if plane_filter is not None:
            plane_name = [k for k, v in self.plane_classes.items() if v == plane_filter][0]
            self.df = self.df[self.df['Plane'] == plane_name]
        
        self.brain_plane_classes = {
            'Not A Brain' : 0, 
            'Trans-thalamic' : 1, 
            'Trans-cerebellum' : 2, 
            'Trans-ventricular' : 3, 
            'Other' : 4
        }
        self.operator_classes = {
            'Other' : 0, 
            'Op. 1' : 1, 
            'Op. 3' : 2, 
            'Op. 2' : 3
        }
        self.machine_classes = {
            'Aloka' : 0, 
            'Other' : 1, 
            'Voluson E6' : 2, 
            'Voluson S10' : 3
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load the image from file
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        # Preprocess and augment the image
        if self.transform:
            image = self.transform(image)
        
        # Return labels for classification task
        plane_cls = self.plane_classes[self.df['Plane'].iloc[idx]]
        brain_plane_cls = self.brain_plane_classes[self.df['Brain_plane'].iloc[idx]]
        operator_cls = self.operator_classes[self.df['Operator'].iloc[idx]]
        machine_cls = self.machine_classes[self.df['US_Machine'].iloc[idx]]
        
        # Return batch
        return image, plane_cls, brain_plane_cls, operator_cls, machine_cls


class GeneratedImageDataset(Dataset):
    """Dataset for generated images."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the generated images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Collect all image paths
        for fname in os.listdir(root_dir):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, fname))
        
        self.image_paths.sort()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def calculate_fid(real_loader, fake_loader, device='cuda'):
    """Calculate FID between real and fake images."""
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Process real images
    print("Processing real images...")
    for batch in tqdm(real_loader):
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        # Ensure images is a tensor
        if not isinstance(images, torch.Tensor):
            continue
            
        # Convert from [-1, 1] to [0, 255] uint8
        images = images.to(device)
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        fid.update(images, real=True)
    
    # Process fake images
    print("Processing fake images...")
    for batch in tqdm(fake_loader):
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        # Ensure images is a tensor
        if not isinstance(images, torch.Tensor):
            continue
            
        # Convert from [-1, 1] to [0, 255] uint8
        images = images.to(device)
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        fid.update(images, real=False)
    
    # Compute FID
    fid_score = fid.compute()
    return fid_score.item()


def main():
    parser = argparse.ArgumentParser(description='Calculate FID for fetal plane dataset')
    parser.add_argument('--real_root', type=str, required=True, help='Root directory of real images')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with annotations')
    parser.add_argument('--fake_root', type=str, required=True, help='Root directory containing generated images')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloaders')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloaders')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device availability
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define transforms for FID calculation (resize to 128x128, normalize to [-1, 1])
    fid_transform = transforms.Compose([
        transforms.Resize(128),  # Resize shortest side to 512
        transforms.CenterCrop(128),  # Center crop to 512x512 (if you want center crop first)
        transforms.RandomHorizontalFlip(p=0.5),  # Mirror around y-axis
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    
    # Class names
    plane_classes = {
        0: 'Other',
        1: 'Maternal cervix',
        2: 'Fetal abdomen',
        3: 'Fetal brain',
        4: 'Fetal femur',
        5: 'Fetal thorax',
    }
    
    results = {}
    
    # Calculate FID for each class
    for class_id, class_name in plane_classes.items():
        fake_dir = os.path.join(args.fake_root, f'diffusion_samples_FETAL_cond_{class_id}')
        
        if not os.path.exists(fake_dir):
            print(f"Warning: Directory {fake_dir} does not exist. Skipping class {class_name}.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Calculating FID for class {class_id}: {class_name}")
        print(f"{'='*60}")
        
        # Load real images for this class
        real_dataset = FetalPlaneDataset(
            root_dir=args.real_root,
            csv_file=args.csv_file,
            transform=fid_transform,
            split=args.split,
            plane_filter=class_id
        )
        
        # Load generated images for this class
        fake_dataset = GeneratedImageDataset(
            root_dir=fake_dir,
            transform=fid_transform
        )
        
        print(f"Real images: {len(real_dataset)}")
        print(f"Fake images: {len(fake_dataset)}")
        
        if len(real_dataset) == 0 or len(fake_dataset) == 0:
            print(f"Skipping class {class_name} due to missing images.")
            continue
        
        # Create dataloaders
        real_loader = DataLoader(
            real_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        fake_loader = DataLoader(
            fake_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Calculate FID
        fid_score = calculate_fid(real_loader, fake_loader, device=device)
        results[class_name] = fid_score
        print(f"FID Score for {class_name}: {fid_score:.4f}")
    
    # Calculate overall FID (all classes combined)
    print(f"\n{'='*60}")
    print("Calculating overall FID (all classes)")
    print(f"{'='*60}")
    
    # Load all real images
    real_dataset_all = FetalPlaneDataset(
        root_dir=args.real_root,
        csv_file=args.csv_file,
        transform=fid_transform,
        split=args.split,
        plane_filter=None
    )
    
    # Collect all fake images from all class directories
    all_fake_paths = []
    for class_id in plane_classes.keys():
        fake_dir = os.path.join(args.fake_root, f'diffusion_samples_FETAL_cond_{class_id}')
        if os.path.exists(fake_dir):
            for fname in os.listdir(fake_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    all_fake_paths.append(os.path.join(fake_dir, fname))
    
    # Create a temporary dataset with all fake images
    class AllFakesDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = sorted(image_paths)
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
    
    fake_dataset_all = AllFakesDataset(all_fake_paths, fid_transform)
    
    print(f"Real images: {len(real_dataset_all)}")
    print(f"Fake images: {len(fake_dataset_all)}")
    
    real_loader_all = DataLoader(
        real_dataset_all,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    fake_loader_all = DataLoader(
        fake_dataset_all,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    overall_fid = calculate_fid(real_loader_all, fake_loader_all, device=device)
    results['Overall'] = overall_fid
    print(f"Overall FID Score: {overall_fid:.4f}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FID SUMMARY")
    print(f"{'='*60}")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")


if __name__ == '__main__':
    main()
