import os
import json
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
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
                - domains: number of domains for lodo cross-validation
                - device: CUDA/CPU
                - log_dir: directory for TensorBoard logs
        """
        self.config = config
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
        self.writer = SummaryWriter("experiments/train_results")
        #self.writer = SummaryWriter("experiments/train_results")
        self.current_domain = 0
        
        self.full_dataset = PACS(
            root=self.config['data_root'],
            test_domain=None
        )
        
    
    def _load_hparams(self, config_path: str) -> Dict:
        """Loads hyperparameters from a yaml config file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Hyperparameter file not found: {config_path}")
    
        with open(config_path) as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            
        if 'best_params' in data:
            return data['best_params']
    
        raise ValueError("Hyperparameter file must contain a 'best_params' section")


    def _init_model(self, hparams: Dict) -> nn.Module:
        """Initialises the best model with given hyperparameters"""
        model = resnet50(
            num_classes=len(self.full_dataset.classes),
            num_domains=len(DOMAIN_NAMES['PACS']),
            batch_size=hparams['batch_size'],
            use_mixstyle=False,
            dropout_p=hparams['dropout'],
            pretrained=True
        )
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
        """Train the model using lodo cross-validation"""
        hparams = self._load_hparams(hparam_path)
        # achtung: lodo splits trainieren jedes Modell für jede einzelne Domain, dh kein spezielles Training
        lodo_splits = get_lodo_splits()
        results = {}
        
        for domain, (train_data, val_data, test_data) in enumerate(lodo_splits):
            print(f"\n=== Fold {domain+1}/{self.config['domains']} ===")
            self.current_domain = domain + 1
            
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

            test_loader = DataLoader(
                Subset(self.full_dataset, test_data),
                batch_size=hparams['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Modell initialisieren
            model = self._init_model(hparams)
            optimizer, scheduler = self._init_optimizer_scheduler(model, hparams)
            criterion = nn.CrossEntropyLoss()

            train_losses = []
            val_losses = []
            val_accuracies = []
            
            # Training loop
            best_val_acc = 0.0
            for epoch in range(self.config['num_epochs']):
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate(model, val_loader, criterion)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # Logging
                self.writer.add_scalar(f'Fold_{domain}/train_loss', train_loss, epoch)
                self.writer.add_scalar(f'Fold_{domain}/val_loss', val_loss, epoch)
                self.writer.add_scalar(f'Fold_{domain}/val_acc', val_acc, epoch)
                
                # Scheduler step
                if scheduler:
                    scheduler.step()
                
                # Bestes Modell speichern
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(model, f"best_fold_{domain}.pt")
                
                print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2%}")
            
            results[f'fold_{domain}'] = {
                'best_val_acc': best_val_acc,
                'final_val_loss': val_loss
            }

            # training curves
            self._plot_training_curves(train_losses, val_losses, val_accuracies, domain)

            # confusion matrix
            self._plot_confusion_matrix(model, test_loader, domain)
            
            # ROC and PR curves (for each class in multi-class)
            self._plot_roc_pr_curves(model, test_loader, domain)
            
            # feature visualization
            self._visualize_embeddings(model, test_loader, domain)
            
            # example predictions
            self._visualize_predictions(model, test_loader, domain, num_examples=5)
        
        # Gesamtergebnisse speichern
        self._plot_comparative_metrics(results)
        self._save_results(results)
        self.writer.close()

        return results


    def _save_model(self, model: nn.Module, filename: str):
        """Save the model state dictionary"""
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': self.current_domain,
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


    # Visualisations    

    def _plot_training_curves(self, train_losses, val_losses, val_accuracies, domain):
        """Plot training and validation metrics over epochs"""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Domain {domain+1}: Training vs Validation Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Domain {domain+1}: Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.config['save_dir'], 'visualizations', f'training_curves_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()
    
    def _plot_confusion_matrix(self, model, loader, domain):
        """Generate and plot confusion matrix"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.full_dataset.classes,
                   yticklabels=self.full_dataset.classes)
        plt.title(f'Domain {domain+1}: Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = os.path.join(self.config['save_dir'], 'visualizations', f'confusion_matrix_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()
    
    def _plot_roc_pr_curves(self, model, loader, domain):
        """Plot ROC and Precision-Recall curves for each class"""
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # One-vs-all for multi-class
        n_classes = len(self.full_dataset.classes)
        plt.figure(figsize=(15, 6))
        
        # ROC curves
        plt.subplot(1, 2, 1)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.full_dataset.classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Domain {domain+1}: ROC Curves')
        plt.legend(loc="lower right")
        
        # PR curves
        plt.subplot(1, 2, 2)
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve((all_labels == i).astype(int), all_probs[:, i])
            plt.plot(recall, precision, label=f'{self.full_dataset.classes[i]}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Domain {domain+1}: Precision-Recall Curves')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        save_path = os.path.join(self.config['save_dir'], 'visualizations', f'roc_pr_curves_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()
    
    def _visualize_embeddings(self, model, loader, domain, n_samples=500):
        """Visualize feature embeddings using t-SNE and UMAP"""
        model.eval()
        features = []
        labels = []
        domains = []
        
        # Get features from penultimate layer
        def hook(module, input, output):
            features.append(output.cpu().numpy())
        
        hook_handle = model.fc.register_forward_hook(hook)
        
        with torch.no_grad():
            for i, (inputs, lbls, doms) in enumerate(loader):
                if len(features) * inputs.shape[0] > n_samples:
                    break
                inputs = inputs.to(self.device)
                _ = model(inputs)
                labels.extend(lbls.cpu().numpy())
                domains.extend(doms.cpu().numpy())
        
        hook_handle.remove()
        features = np.concatenate(features)[:n_samples]
        labels = np.array(labels)[:n_samples]
        domains = np.array(domains)[:n_samples]
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features)
        
        plt.figure(figsize=(12, 5))
        
        # By class
        plt.subplot(1, 2, 1)
        for i in range(len(self.full_dataset.classes)):
            mask = labels == i
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                       label=self.full_dataset.classes[i], alpha=0.6)
        plt.title(f'Domain {domain+1}: t-SNE by Class')
        plt.legend()
        
        # By domain
        plt.subplot(1, 2, 2)
        for i in range(len(DOMAIN_NAMES['PACS'])):
            mask = domains == i
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                       label=DOMAIN_NAMES['PACS'][i], alpha=0.6)
        plt.title(f'Domain {domain+1}: t-SNE by Domain')
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.config['save_dir'], 'visualizations', f'tsne_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()
        
        # UMAP visualization (similar code as t-SNE)
        # [UMAP implementation would go here]
    
    def _visualize_predictions(self, model, loader, domain, num_examples: int = 5):
        """Visualize example predictions with images"""
        model.eval()
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, num_examples*3))
        
        for i, (inputs, labels, _) in enumerate(loader):
            if i >= num_examples:
                break
                
            inputs = inputs.to(self.device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Original image
            img = inputs[0].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Original\nTrue: {self.full_dataset.classes[labels[0]]}')
            axes[i, 0].axis('off')
            
            # Prediction probabilities
            axes[i, 1].barh(self.full_dataset.classes, probs[0].cpu().numpy())
            axes[i, 1].set_title(f'Prediction Probabilities\nPred: {self.full_dataset.classes[preds[0]]}')
            axes[i, 1].set_xlim(0, 1)
            
            # Saliency map (simple version)
            inputs.requires_grad_()
            outputs = model(inputs)
            loss = outputs[0, preds[0]]
            loss.backward()
            
            saliency = inputs.grad.data.abs().max(dim=1)[0][0].cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            
            axes[i, 2].imshow(img)
            axes[i, 2].imshow(saliency, cmap='hot', alpha=0.5)
            axes[i, 2].set_title('Saliency Map')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.config['save_dir'], 'visualizations', f'predictions_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()
    
    def _plot_comparative_metrics(self, results):
        """Plot comparative metrics across domains"""
        domains = list(results.keys())
        accuracies = [results[d]['best_val_acc'] for d in domains]
        
        plt.figure(figsize=(8, 5))
        plt.bar(domains, accuracies)
        plt.xlabel('Domain')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Across Domains')
        plt.ylim(0, 1)
        
        save_path = os.path.join(self.config['save_dir'], 'visualizations', 'comparative_metrics.png')
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Framework')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--hparam_file', type=str, required=True, help='Path to best hyperparameters')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--domains', type=int, default=4, help='Number of domains')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # Konfiguration
    config = {
        'data_root': "/mnt/data/hahlers/datasets/PACS", #"experiments/train_results",
        'hparam_file': "configs/global_config.yaml",
        'num_epochs': args.num_epochs,
        'domains': args.domains,
        'device': args.device,
        'mode': args.mode,
        'log_dir': "experiments/train_results/logs/training",
        'save_dir': "experiments/train_results/saved_models"
    }
    
    # Verzeichnisse erstellen
    #os.makedirs(config['data_root'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)

    # Training starten
    trainer = TrainingFramework(config)
    results = trainer.run(args.hparam_file)
    
    print("\n=== Training Complete ===")
    print(f"Average Validation Accuracy: {results['average_val_acc']:.2%}")