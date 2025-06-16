import os
import json
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from data._load_data import get_lodo_splits
from models import resnet50
from data._datasets import PACS, DOMAIN_NAMES

class TrainingFramework:
    def __init__(self, config: Dict):
        """
        Training framework with lodo cross-validation for image classification tasks
        
        Args:
            config: Configuration dictionary with
                - data_root: path to dataset
                - batch_size: batch size
                - num_epochs: number of epochs
                - folds: number of folds for lodo cross-validation
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
        )
        
    # TODO
    def _load_hparams(self, hparam_path: str) -> Dict:
        """Loads hyperparameters from a JSON file, optionally for a specific domain"""
        if not os.path.exists(hparam_path):
            raise FileNotFoundError(f"Hyperparameter file not found: {hparam_path}")
        
        with open(hparam_path) as f:
            data = json.load(f)
            
        if domain is not None:
            # Find parameters for specific domain
            for entry in data['domain_specific']:
                if entry['domain'] == domain:
                    return entry['params']
            raise ValueError(f"No parameters found for domain {domain}")
        
        # Return global parameters
        return data['global']['params']


    #TODO soll modell mit mixstyle initialisieren wenn es so als parameter übergeben wird, 
    # ansonsten ohne mixstyle
    def _init_model(self, hparams: Dict) -> nn.Module:
        """Initialises the best model with given hyperparameters"""
        model = resnet50(
            num_classes=len(self.full_dataset.classes),
            num_domains=len(DOMAIN_NAMES['PACS']),
            batch_size=hparams['batch_size'],
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
        if hparams['optimizer'] == 'AdamW' or hparams['optimizer'] == 'Adam':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hparams['lr'],
                weight_decay=hparams['weight_decay'],
                betas=(hparams['beta1'], hparams['beta2']),
            )
        elif hparams['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hparams['lr'],
                momentum=hparams.get('momentum', 0.9),
                weight_decay=hparams['weight_decay'],
                nesterov=hparams.get('nesterov', False),
            )
        
        if hparams['scheduler'] == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['num_epochs'], #T_max= hparams['T_max'],
                eta_min=hparams['eta_min']
            )

        if hparams['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hparams['step_size'],
                gamma=hparams['gamma']
            )
        return optimizer, scheduler


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
        # achtung: lodo splits trainieren jedes Modell für jede einzelne Domain, dh kein spezielles Training
        lodo_splits = get_lodo_splits()
        results = {}
        
        for fold, (train_data, val_data, test_data) in enumerate(lodo_splits):
            print(f"\n=== Fold {fold+1}/{self.config['k_folds']} ===")
            self.current_fold = fold + 1
            
            # Datenaufteilung
            train_set = Subset(self.full_dataset, train_data)
            val_set = Subset(self.full_dataset, val_data)
            
            train_loader = DataLoader(
                train_set,
                batch_size=hparams['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_set,
                batch_size=hparams['batch_size'],
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


#TODO wenn --mode=domain, dann müssen vier Modelle trainiert werden, die automatisch die richtigen yamls
# als grundlage benutzen
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Framework')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--hparam_file', type=str, required=True, help='Path to best hyperparameters')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--folds', type=int, default=4, help='Number of folds')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='global', choices=['global', 'domain'])
    args = parser.parse_args()

    # Konfiguration
    config = {
        'data_root': "experiments/train_results",
        'num_epochs': args.num_epochs,
        'folds': args.folds,
        'device': args.device,
        'mode': args.mode,
        'log_dir': 'logs/training',
        'save_dir': 'saved_models'
    }
    
    # Verzeichnisse erstellen
    os.makedirs(config['data_root'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)

    # Training starten
    trainer = TrainingFramework(config)
    results = trainer.run(args.hparam_file)
    
    print("\n=== Training Complete ===")
    print(f"Average Validation Accuracy: {results['average_val_acc']:.2%}")