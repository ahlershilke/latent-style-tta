import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from datetime import datetime
from typing import Dict, List, Tuple
from models import resnet50
from data._datasets import PACS, DOMAIN_NAMES

class TrainingFramework:
    def __init__(self, config: Dict):
        """
        Training framework with k-fold cross-validation for image classification tasks
        
        Args:
            config: Configuration dictionary with
                - data_root: path to dataset
                - batch_size: batch size
                - num_epochs: number of epochs
                - k_folds: number of folds for k-fold cross-validation
                - device: CUDA/CPU
                - log_dir: directory for TensorBoard logs
        """
        self.config = config
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")) #self.device = torch.device(config['device'])
        self.writer = SummaryWriter("experiments/train_results")
        # self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.current_fold = 0
        
        self.full_dataset = PACS(
            root=config['data_root'],
            test_domain=None
            #augment=self._get_augmentations()
        )
        

    """
    def _get_augmentations(self):
        #Industriestandard Augmentations für Bilddaten
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    """


    def _load_best_hparams(self, hparam_path: str) -> Dict:
        """Loads the best hyperparameters from a JSON file"""
        if not os.path.exists(hparam_path):
            raise FileNotFoundError(f"Hyperparameter file not found: {hparam_path}")
        
        with open(hparam_path) as f:
            return json.load(f)['best_params']


    def _init_model(self, hparams: Dict) -> nn.Module:
        """Initialises the best model with given hyperparameters"""
        model = resnet50(
            num_classes=len(self.full_dataset.classes),
            num_domains=len(DOMAIN_NAMES['PACS']),
            use_mixstyle=True,
            mixstyle_layers=hparams['mixstyle_layers'].split('+'),
            mixstyle_p=hparams['mixstyle_p'],
            mixstyle_alpha=hparams['mixstyle_alpha'],
            dropout_p=hparams['dropout'],
            pretrained=True
        )
        return model.to(self.device)


    def _init_optimizer(self, model: nn.Module, hparams: Dict) -> Tuple:
        """Initialises the optimizer and learning rate scheduler"""
        if hparams['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hparams['lr'],
                weight_decay=hparams['weight_decay']
            )
        elif hparams['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hparams['lr'],
                momentum=hparams.get('momentum', 0.9),
                weight_decay=hparams['weight_decay']
            )
        
        if hparams['scheduler'] == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['num_epochs']
            )
        return optimizer, scheduler


    def _kfold_split(self) -> List[Tuple]:
        """Generates k-fold splits for cross-validation"""
        kf = KFold(n_splits=self.config['k_folds'], shuffle=True)
        return list(kf.split(range(len(self.full_dataset))))


    def train_epoch(self, model: nn.Module, loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """One epoch of training"""
        model.train()
        total_loss = 0.0
        
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)


    def validate(self, model: nn.Module, loader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float]:
        """Validation phase"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return val_loss / len(loader), accuracy


    def run(self, hparam_path: str):
        """Train the model using k-fold cross-validation"""
        hparams = self._load_best_hparams(hparam_path)
        kfold_splits = self._kfold_split()
        results = {}
        
        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"\n=== Fold {fold+1}/{self.config['k_folds']} ===")
            self.current_fold = fold + 1
            
            # Datenaufteilung
            train_set = Subset(self.full_dataset, train_idx)
            val_set = Subset(self.full_dataset, val_idx)
            
            train_loader = DataLoader(
                train_set,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_set,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Modell initialisieren
            model = self._init_model(hparams)
            optimizer, scheduler = self._init_optimizer(model, hparams)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_acc = 0.0
            for epoch in range(self.config['num_epochs']):
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate(model, val_loader, criterion)
                
                # Logging
                self.writer.add_scalar(f'Fold_{fold}/train_loss', train_loss, epoch)
                self.writer.add_scalar(f'Fold_{fold}/val_loss', val_loss, epoch)
                self.writer.add_scalar(f'Fold_{fold}/val_acc', val_acc, epoch)
                
                # Scheduler step
                if scheduler:
                    scheduler.step()
                
                # Bestes Modell speichern
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(model, f"best_fold_{fold}.pt")
                
                print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2%}")
            
            results[f'fold_{fold}'] = {
                'best_val_acc': best_val_acc,
                'final_val_loss': val_loss
            }
        
        # Gesamtergebnisse speichern
        self._save_results(results)
        self.writer.close()
        return results


    def _save_model(self, model: nn.Module, filename: str):
        """Save the model state dictionary"""
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': self.current_fold,
            'timestamp': datetime.now().isoformat()
        }, save_path)


    def _save_results(self, results: Dict):
        """Saves the training results to a JSON file"""
        result_path = os.path.join(self.config['save_dir'], 'kfold_results.json')
        with open(result_path, 'w') as f:
            json.dump({
                'config': self.config,
                'results': results,
                'average_val_acc': sum(r['best_val_acc'] for r in results.values()) / len(results)
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Industrial Training Framework')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--hparam_file', type=str, required=True, help='Path to best hyperparameters')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of k-folds')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # Konfiguration
    config = {
        'data_root': args.data_root,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'k_folds': args.k_folds,
        'device': args.device,
        'log_dir': 'logs/industrial_training',
        'save_dir': 'saved_models'
    }
    
    # Verzeichnisse erstellen
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)

    # Training starten
    trainer = TrainingFramework(config)
    results = trainer.run_kfold_training(args.hparam_file)
    
    print("\n=== Training Complete ===")
    print(f"Average Validation Accuracy: {results['average_val_acc']:.2%}")