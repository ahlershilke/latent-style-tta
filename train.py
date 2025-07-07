import os
import json
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import subprocess
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from data._load_data import get_lodo_splits
from models import resnet50, resnet18
from data._datasets import PACS, DOMAIN_NAMES, DomainDataset, DomainSubset
from utils._visualize import Visualizer
from utils._utils import save_training_results
from models._styleextraction import StyleExtractorManager


class TrainingFramework:
    def __init__(
            self,
            config: Dict,  
            dataset: DomainDataset,
            class_names: List[str],
            domain_names: List[str]
        ):
        """
        Training framework with lodo cross-validation for image classification tasks
        
        Args:
            config: Configuration dictionary with
                - data_root: path to dataset
                - batch_size: batch size
                - num_epochs: number of epochs
                - domains: number of domains for lodo cross-validation
                - device: CUDA/CPU
                - log_dir: directory for TensorBoard logs
        """

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(self.config['log_dir'])
        
        # Datenverwaltung
        self.full_dataset = dataset
        self.class_names = class_names
        self.domain_names = domain_names
        self.lodo_splits = self.full_dataset.generate_lodo_splits()

        self.collate_fn = self.full_dataset.collate_fn if hasattr(self.full_dataset, 'collate_fn') else default_collate
        
        self.visualizer = Visualizer(
            config=self.config,
            class_names=self.class_names,
            domain_names=self.domain_names,
            vis_dir=self.config['vis_dir']
        )
        
        self.style_manager = StyleExtractorManager(
            domain_names=self.domain_names,
            device=self.device
        )


    def set_seeds(seed: int = 42) -> None:
        """Sets all random seeds for reproducibility.
    
        Args:
            seed: Random seed (default: 42).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    
    def _load_hparams(self, config_path: str) -> Dict:
        """Loads hyperparameters from a yaml config file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Hyperparameter file not found: {config_path}")
    
        with open(config_path) as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
                return data
            
        raise ValueError("Could not load hyperparameters. Please provide a valid YAML file.")


    def _init_model(self, hparams: Dict) -> nn.Module:
        """Initialises the best model with given hyperparameters"""
        model = resnet50(
            num_classes=len(self.class_names),
            num_domains=len(DOMAIN_NAMES['PACS']),
            batch_size=hparams['batch_size'],
            use_mixstyle=False,
            dropout_p=hparams['dropout'],
            pretrained=True
        )
        model.enable_style_stats(True)
        return model.to(self.device)


    def _init_optimizer_scheduler(self, model: nn.Module, hparams: Dict) -> Tuple:
        """Initialises the optimizer and learning rate scheduler"""
        if hparams['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hparams['lr'],
                weight_decay=hparams['weight_decay'],
                betas=(hparams['beta1'], hparams['beta2']),
                eps=hparams.get('eps', 1e-8)
            )
        elif hparams['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hparams['lr'],
                weight_decay=hparams['weight_decay'],
                betas=(hparams['beta1'], hparams['beta2']),
                eps=hparams.get('eps', 1e-8)
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
                T_max=hparams['T_max'], #T_max=self.config['num_epochs'], #T_max= hparams['T_max'],
                eta_min=hparams['eta_min']
            )
        elif hparams['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hparams['step_size'],
                gamma=hparams['gamma']
            )
        elif hparams['scheduler'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=hparams['factor'],
                patience=hparams['patience'],
                verbose=True
            )

        return optimizer, scheduler
    

    def extract_style_stats_from_saved_models(self, hparam_path: str) -> None:
        """Extracts style stats from all saved model checkpoints
        hparams = self._load_hparams(hparam_path)
        
        for domain_idx, domain_name in enumerate(self.domain_names):
            model_path = os.path.join(self.config['save_dir'], f"best_fold_{domain_name}.pt")
            
            if not os.path.exists(model_path):
                print(f"Warning: Model checkpoint not found for {domain_name}")
                continue
            
            print(f"\nExtracting style stats from {domain_name} model...")
            
            self.style_manager.extract_from_saved_model(
                model_path=model_path,
                domain_name=domain_name,
                model_class=resnet50,
                model_args={
                    'num_classes': len(self.class_names),
                    'num_domains': len(self.domain_names),
                    'batch_size': hparams['batch_size'],
                    'use_mixstyle': False,
                    'dropout_p': hparams['dropout'],
                    'pretrained': True
                },
                results_dir=os.path.join(self.config['save_dir'], "style_stats")
            )
        """
        """Extracts style stats from all trained models"""
        hparams = self._load_hparams(hparam_path)

        for domain_idx, domain_name in enumerate(self.domain_names):
            model_path = os.path.join(self.config['save_dir'], f"best_fold_{domain_name}.pt")

            if not os.path.exists(model_path):
                continue
            
            # Load model with style stats
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model = self._init_model(hparams)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
            if 'style_stats' in checkpoint:
                model.style_stats.load_state_dict(checkpoint['style_stats'], strict=False)
        
            # Save stats
            self.style_manager.save_style_stats(
                domain_name=domain_name,
                results_dir=os.path.join(self.config['save_dir'], "style_stats")
            )


    def train_epoch(self, model: nn.Module, loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """One epoch of training"""
        model.train()
        model.enable_style_stats(True)
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, domains in loader:
            inputs, labels, domains = inputs.to(self.device), labels.to(self.device), domains.to(self.device)
        
            optimizer.zero_grad()
            outputs = model(inputs, domain_idx=domains)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
        return total_loss / len(loader), accuracy


    def validate(self, model: nn.Module, loader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float]:
        """Validation phase"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, domains in loader:
                inputs, labels, domains = inputs.to(self.device), labels.to(self.device), domains.to(self.device)

                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
            
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
        return val_loss / len(loader), accuracy


    def run(self, hparam_path: str):
        """Train the model using lodo cross-validation"""
        #self.visualizer._visualize_raw_dataset(loader=DataLoader(self.full_dataset, batch_size=32, shuffle=False), n_samples=500)
        #self.visualizer._visualize_complete_raw_dataset(loader=DataLoader(self.full_dataset, batch_size=32, shuffle=False))
        hparams = self._load_hparams(hparam_path)
        results = {
            'all_val_acc': [],
            'all_train_loss': [],
            'all_val_loss': [],
            'all_train_acc': [],
            'per_domain': {}
        }

        all_true_labels, all_pred_labels, all_domains = [], [], []
    
        # Nutze die intern generierten Splits statt get_lodo_splits()
        domain_pbar = tqdm(
            enumerate(self.lodo_splits), 
            total=len(self.lodo_splits), 
            desc="Domains"
        )
    
        for domain_idx, (train_data, val_data, test_data) in domain_pbar:
            domain_name = self.domain_names[domain_idx] if domain_idx < len(self.domain_names) else f"Domain_{domain_idx+1}"
            domain_pbar.set_description(f"Domain {domain_idx} ({domain_name})")
            self.current_domain = domain_idx + 1
        
            #TODO sicher, dass die loader hier im loop initialisiert werden??
            # DataLoader mit konsistenten Domain-Indizes
            train_loader = DataLoader(
                train_data,
                batch_size=hparams['batch_size'],
                shuffle=True,
                pin_memory=True,
                collate_fn=self.collate_fn
            )

            val_loader = DataLoader(
                val_data,
                batch_size=hparams['batch_size'],
                shuffle=False,
                pin_memory=True,
                collate_fn=self.collate_fn
            )

            test_loader = DataLoader(
                test_data,  # test_data ist bereits ein DomainSubset mit korrektem domain_idx
                batch_size=hparams['batch_size'],
                shuffle=False,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        
            # Modell und Training (unverändert)
            model = self._init_model(hparams)
            optimizer, scheduler = self._init_optimizer_scheduler(model, hparams)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            best_epoch_stats = None
            epoch_train_losses, epoch_train_accs = [], []
            epoch_val_losses, epoch_val_accs = [], []
            epoch_test_losses, epoch_test_accs = [], []
        
            # Training Loop (unverändert)
            epoch_pbar = tqdm(range(self.config['num_epochs']), desc="Epochs", leave=False)
            for epoch in epoch_pbar:
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate(model, val_loader, criterion)
                test_loss, test_acc = self.validate(model, test_loader, criterion)

                epoch_train_losses.append(train_loss)
                epoch_train_accs.append(train_acc)
                epoch_val_losses.append(val_loss)
                epoch_val_accs.append(val_acc)
                epoch_test_losses.append(test_loss)
                epoch_test_accs.append(test_acc)
            
                # Logging
                self.writer.add_scalar(f'Fold_{domain_idx}/train_loss', train_loss, epoch)
                self.writer.add_scalar(f'Fold_{domain_idx}/val_loss', val_loss, epoch)
                self.writer.add_scalar(f'Fold_{domain_idx}/val_acc', val_acc, epoch)
                self.writer.add_scalar(f'Fold_{domain_idx}/test_acc', test_acc, epoch)
            
                if scheduler:
                    scheduler.step()
            
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(model, f"best_fold_{domain_name}.pt")
                    best_epoch_stats = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'test_acc': test_acc,
                        'epoch': epoch
                    }
            
                epoch_pbar.set_postfix({
                    'train_loss': f"{train_loss:.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'val_acc': f"{val_acc:.2%}",
                    'test_acc': f"{test_acc:.2%}"
                })

            results['per_domain'][f'domain_{domain_idx}'] = {
                'name': domain_name,
                'best_epoch': best_epoch_stats,
                'epoch_train_losses': epoch_train_losses,
                'epoch_val_losses': epoch_val_losses,
                'epoch_val_accs': epoch_val_accs,
                'epoch_test_accs': epoch_test_accs
            }

            results['all_val_acc'].append(best_val_acc)
            results['all_train_loss'].append(best_epoch_stats['train_loss'])
            results['all_val_loss'].append(best_epoch_stats['val_loss'])
            results['all_train_acc'].append(best_epoch_stats['test_acc'])
            
            test_preds, test_labels, test_domains = self.visualizer._evaluate_and_collect(model, test_loader)
            all_true_labels.extend(test_labels)
            all_pred_labels.extend(test_preds)
            all_domains.extend(test_domains)

            """
            loaders_dict = {
                domain_name: DataLoader(
                    DomainSubset(test_data, indices=range(len(test_data)), domain_idx=idx),
                    batch_size=hparams['batch_size'],
                    shuffle=False
                )
                for idx, (domain_name, test_data) in enumerate(zip(self.domain_names, test_data))
            }
            """

            dl = DataLoader(
                dataset=self.full_dataset,
                batch_size=64,
                shuffle=False,
                pin_memory=True
                )

            # Visualisierungen
            #self.visualizer.visualize_resnet_tsne_blocks(
                #model=model,
                #loader=dl,
                #device=self.device,
                #n_samples=500,
                #block_names=['layer1', 'layer2', 'layer3', 'layer4']
            #)
            #self.visualizer._plot_training_curves(epoch_train_losses, epoch_val_losses, epoch_test_losses, epoch_train_accs, epoch_val_accs, epoch_test_accs, domain_name)
            #self.visualizer._plot_roc_pr_curves(model, test_loader, domain_name)
            #self.visualizer._plot_confusion_matrix(model, test_loader, domain_name)
            #self.visualizer._visualize_full_embedded_dataset(model, loader=DataLoader(self.full_dataset, batch_size=32, shuffle=False))
            #self.visualizer._visualize_predictions(model, test_loader, domain_name, num_examples=5)
            #self.visualizer._visualize_gradcam_predictions(
                #model=model,
                #loader=test_loader,
                #domain_name=domain_name,
                #num_examples=5,
                #target_layer=None  # will auto-detect last conv layer
            #)
            #self.visualizer._visualize_umap_embeddings(
                #model=model,
                #loader=test_loader,  # oder DataLoader(self.full_dataset)
                #domain_name=domain_name,
                #n_samples=500
            #)
            #self.visualizer._visualize_raw_umap(loader=test_loader, domain_name=domain_name)
    
        results['avg_val_acc'] = np.mean(results['all_val_acc'])
        results['avg_train_loss'] = np.mean(results['all_train_loss'])
        results['avg_val_loss'] = np.mean(results['all_val_loss'])
        results['avg_test_acc'] = np.mean(results['all_train_acc'])

        # Gesamtergebnisse
        #self.visualizer._plot_comparative_metrics(results)
        #self.visualizer._visualize_embedded_dataset(
            #model=model,
            #loader=DataLoader(self.full_dataset, batch_size=32, shuffle=False),
            #n_samples=500
        #)
        #self._save_results(results)
        self.writer.close()

        self.extract_style_stats_from_saved_models(hparam_path)

        return results

    def _save_model(self, model: nn.Module, filename: str):
        """Save the model state dictionary"""
        save_path = os.path.join(self.config['save_dir'], filename)
        
        model_state_dict = {
            k: v for k, v in model.state_dict().items() 
            if not k.startswith('style_stats.')
        }
        
        torch.save({
            'model_state_dict': model_state_dict,
            'style_stats': model.style_stats.state_dict(),
            'style_stats_config': model.style_stats_config,
            'fold': self.current_domain,
            'config': self.config,
            'git_hash': subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            'timestamp': datetime.now().isoformat()
        }, save_path)

    
    def _save_results(self, results: Dict):
        """Saves the training results to a JSON file"""
        result_path = os.path.join(self.config['save_dir'], 'lodo_results.json')
    
        # Berechne average_val_acc aus den gespeicherten Werten
        avg_val_acc = results['avg_val_acc'] if 'avg_val_acc' in results else \
                     sum(results['all_val_acc']) / len(results['all_val_acc'])

        with open(result_path, 'w') as f:
            json.dump({
                'config': self.config,
                'results': results,
                'average_val_acc': avg_val_acc
            }, f, indent=2)


if __name__ == "__main__":
    TrainingFramework.set_seeds(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/mnt/data/hahlers/datasets")
    parser.add_argument('--hparam_file', type=str, default="configs/global_config.yaml")
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--domains', type=int, default=4, help='Number of domains')
    args = parser.parse_args()

    # Konfiguration
    config = {
        'data_root': args.data_root if args.data_root else "/mnt/data/hahlers/datasets",
        'hparam_file': args.hparam_file if args.hparam_file else "/mnt/data/hahlers/tuning/configs/global_config.yaml",
        'num_epochs': args.num_epochs,
        'domains': args.domains,
        'log_dir': "experiments/train_results/logs",
        'save_dir': "experiments/train_results/saved_models",
        'vis_dir': "experiments/train_results/visualizations"
    }
    
    # Verzeichnisse erstellen
    #os.makedirs(config['data_root'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['vis_dir'], exist_ok=True)

    full_dataset = PACS(root=config['data_root'], test_domain=None)

    trainer = TrainingFramework(
        config=config,
        dataset=full_dataset,
        class_names=full_dataset.classes,
        domain_names=DOMAIN_NAMES["PACS"]
    )
    
    hparam_path = args.hparam_file if args.hparam_file else config['hparam_file']
    results = trainer.run(args.hparam_file)
    
    print("\n=== Training Complete ===")
    print(f"Average Validation Accuracy: {results['avg_val_acc']:.2%}")

    save_training_results(config, "/mnt/data/hahlers/training")