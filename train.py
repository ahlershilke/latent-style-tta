import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from models._resnet import resnet18, resnet34, resnet50, resnet101
from models._mixstyle import MixStyle
from data._datasets import PACS
from torch.utils.data import Subset, ConcatDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Domain Generalization with MixStyle')
    parser.add_argument('--data_root', type=str, required=True, 
                       help='Root directory of dataset')
    parser.add_argument('--test_domain', type=int, default=None,
                       help='Domain index to use as test domain (leave-one-out)')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Input batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='Number of classes')
    parser.add_argument('--num_domains', type=int, default=4,
                       help='Number of domains')
    parser.add_argument('--mixstyle_p', type=float, default=0.5,
                       help='Probability of applying MixStyle')
    parser.add_argument('--mixstyle_alpha', type=float, default=0.3,
                       help='Beta distribution alpha parameter for MixStyle')
    parser.add_argument('--mixstyle_layers', type=str, default='layer1,layer2',
                       help='Comma-separated list of layers to apply MixStyle')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory to save outputs')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with small subset of data')
    return parser.parse_args()

def setup_environment(args):
    """Create output directories and setup device"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'stats'), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return device

def get_model(args, device):
    """Initialize model with MixStyle"""
    model_fn = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101
    }[args.model]
    
    mixstyle_layers = args.mixstyle_layers.split(',')
    
    model = model_fn(
        num_classes=args.num_classes,
        num_domains=args.num_domains,
        use_mixstyle=True,
        mixstyle_layers=mixstyle_layers,
        mixstyle_p=args.mixstyle_p,
        mixstyle_alpha=args.mixstyle_alpha
    ).to(device)
    
    return model

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}', dynamic_ncols=True)
    for images, class_labels, domain_labels in pbar:
        images = images.to(device)
        class_labels = class_labels.to(device)
        domain_labels = domain_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, domain_labels)
        
        loss = criterion(outputs, class_labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += class_labels.size(0)
        correct += (predicted == class_labels).sum().item()
        running_loss += loss.item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate model performance"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, class_labels, domain_labels in tqdm(loader, desc='Validation'):
            images = images.to(device)
            class_labels = class_labels.to(device)
            domain_labels = domain_labels.to(device)
            
            outputs = model(images, domain_labels)
            loss = criterion(outputs, class_labels)
            
            _, predicted = torch.max(outputs, 1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()
            running_loss += loss.item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def save_checkpoint(model, optimizer, epoch, args, best=False):
    """Save model checkpoint"""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args)
    }
    
    filename = f'checkpoint_{epoch}.pth' if not best else 'best_checkpoint.pth'
    torch.save(state, os.path.join(args.output_dir, 'checkpoints', filename))

def main():
    args = parse_args()
    device = setup_environment(args)
    
    # Initialize dataset and loaders
    full_dataset = PACS(root=args.data_root, test_domain=args.test_domain, augment=None)
    
    if args.debug:
        debug_size = min(500, int(0.1 * len(full_dataset)))
        dataset = Subset(full_dataset, range(debug_size))
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader, val_loader, _ = full_dataset.generate_loaders(
            batch_size=args.batch_size,
            test_size=0.2,
            stratify=True
        )
    
    # Initialize model, loss and optimizer
    model = get_model(args, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, args)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, args, best=True)
        
        # Save style statistics
        stats_path = os.path.join(args.output_dir, 'stats', f'style_stats_{epoch}.json')
        model.get_style_stats().save_style_stats_to_json(stats_path)
        
        print(f'Epoch {epoch}/{args.epochs}: '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    
    # Save final model and training history
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()