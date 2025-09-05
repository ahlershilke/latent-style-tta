import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import umap.umap_ as umap
from typing import List, Dict, DefaultDict, Optional
from data._datasets import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import torch
from data._datasets import DOMAIN_NAMES


class Visualizer:
    def __init__(self, config: Optional[dir], class_names, domain_names, vis_dir: Optional[dir]): #, dataset):
        self.config = config if config is not None else {}
        self.class_names = class_names
        self.domain_names = domain_names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vis_dir = vis_dir

        subdirs = [
            "training_curves",
            "confusion_matrices", 
            "roc_pr",
            "heatmaps",
            "tsne",
            "umap",
            "test_stats"
        ]

        for subdir in subdirs:
            dir_path = os.path.join(self.vis_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)


    def _plot_training_curves(self, train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, domain):
        """Plot training and validation metrics over epochs"""
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Domain {domain}: Training vs Validation vs Test Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')
        plt.plot(test_accuracies, label='Test Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Domain {domain}: Training, Validation and Test Accuracy')
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "training_curves", f'training_curves_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()

    
    def calculate_test_statistics(self, all_results):
        """
        Berechnet Worst-Case, Best-Case und Average-Case Test-Accuracies
        
        Args:
            all_results: Liste von Ergebnissen aus mehreren Seeds
            
        Returns:
            Dictionary mit berechneten Statistiken
        """
        # Test-Accuracies über alle Domains und Seeds sammeln
        all_test_accuracies = []
        per_domain_test_accuracies = {domain: [] for domain in self.domain_names}

        for result in all_results:
            # Gesamt-Test-Accuracy pro Seed
            if 'avg_test_acc' in result:
                all_test_accuracies.append(result['avg_test_acc'])
            
            # Domain-spezifische Test-Accuracies
            for domain_idx, domain_name in enumerate(self.domain_names):
                domain_key = f'domain_{domain_idx}'
                if domain_key in result['per_domain'] and 'best_epoch' in result['per_domain'][domain_key]:
                    domain_test_acc = result['per_domain'][domain_key]['best_epoch']['test_acc']
                    if domain_test_acc is not None:
                        per_domain_test_accuracies[domain_name].append(domain_test_acc)

        valid_domains = {k: v for k, v in per_domain_test_accuracies.items() if len(v) > 0}
        # Statistik-Berechnung
        test_stats = {
            'average_case': {
                'mean': np.mean(all_test_accuracies) if all_test_accuracies else 0,
                'std': np.std(all_test_accuracies) if all_test_accuracies else 0,
                'min': np.min(all_test_accuracies) if all_test_accuracies else 0,
                'max': np.max(all_test_accuracies) if all_test_accuracies else 0,
                'all_values': all_test_accuracies
            },
            'best_case': {
                'mean': np.mean([max(accs) for accs in valid_domains.values()]) if valid_domains else 0,
                'std': np.std([max(accs) for accs in valid_domains.values()]) if valid_domains else 0,
                'all_values': [max(accs) for accs in valid_domains.values()] if valid_domains else []
            },
            'worst_case': {
                'mean': np.mean([min(accs) for accs in valid_domains.values()]) if valid_domains else 0,
                'std': np.std([min(accs) for accs in valid_domains.values()]) if valid_domains else 0,
                'all_values': [min(accs) for accs in valid_domains.values()] if valid_domains else []
            },
            'per_domain': {
                domain: {
                    'mean': np.mean(accs) if accs else 0,
                    'std': np.std(accs) if accs else 0,
                    'min': np.min(accs) if accs else 0,
                    'max': np.max(accs) if accs else 0,
                    'all_values': accs
                }
                for domain, accs in per_domain_test_accuracies.items()
            }
        }

        # Visualisierung der Ergebnisse
        self._plot_test_statistics(test_stats)
        #self._save_test_statistics(test_stats)
        
        return test_stats

    def _plot_test_statistics(self, test_stats):
        """Erstellt Visualisierungen der Test-Statistiken"""
        
        # Boxplot für alle Test-Accuracies
        plt.figure(figsize=(12, 6))
        
        # Daten vorbereiten
        data = [
            test_stats['average_case']['all_values'],
            test_stats['best_case']['all_values'],
            test_stats['worst_case']['all_values']
        ]
        
        labels = ['Average Case', 'Best Case', 'Worst Case']
        
        # Boxplot erstellen
        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.title('Test Accuracy Distribution Across Cases')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Speichern
        save_path = os.path.join(self.vis_dir, "test_stats", "test_cases_boxplot.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Per-Domain Visualisierung
        plt.figure(figsize=(12, 6))
        
        domain_means = [stats['mean'] for stats in test_stats['per_domain'].values()]
        domain_stds = [stats['std'] for stats in test_stats['per_domain'].values()]
        domain_names = list(test_stats['per_domain'].keys())
        
        plt.bar(domain_names, domain_means, yerr=domain_stds, capsize=5)
        plt.title('Test Accuracy Per Test Domain')
        plt.ylabel('Mean Accuracy')
        plt.ylim(0, 1)
        
        save_path = os.path.join(self.vis_dir, "test_stats", "per_domain_accuracy.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_accuracy_development(self, all_results, metric='val_acc'):
        """
        Plots the development of average, best, and worst case accuracies across seeds.

        Args:
            all_results: List of result dictionaries from multiple seeds
            metric: 'val_acc' or 'test_acc' to plot validation or test accuracy
        title: Custom plot title
            save_path: Optional path to save the figure
        """
        # Determine max number of epochs across all runs
        max_epochs = max(
            len(run['per_domain'][domain][f'epoch_{metric}s'])
            for run in all_results
            for domain in run['per_domain']
        )
    
        # Initialize containers
        all_accuracies = np.zeros((len(all_results), len(self.domain_names), max_epochs))
    
        # Fill with data (pad shorter runs with their last value)
        for seed_idx, run in enumerate(all_results):
            for domain_idx, domain_data in enumerate(run['per_domain'].values()):
                accs = domain_data[f'epoch_{metric}s']
                padded_accs = accs + [accs[-1]] * (max_epochs - len(accs))
                all_accuracies[seed_idx, domain_idx] = padded_accs
    
        # Calculate statistics
        avg_acc = np.mean(all_accuracies, axis=(0, 1))  # Average across seeds and domains
        best_acc = np.max(all_accuracies, axis=(0, 1))  # Best case
        worst_acc = np.min(all_accuracies, axis=(0, 1))  # Worst case
    
        # Plot
        plt.figure(figsize=(10, 6))
        epochs = np.arange(1, max_epochs + 1)
    
        plt.plot(epochs, avg_acc, label='Average Case', color='blue')
        plt.plot(epochs, best_acc, label='Best Case', linestyle='--', color='green')
        plt.plot(epochs, worst_acc, label='Worst Case', linestyle='--', color='red')
    
        plt.fill_between(epochs, best_acc, worst_acc, color='gray', alpha=0.1)

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{metric} Development Across Seeds')
        plt.legend()

        save_path = os.path.join(self.vis_dir, "test_stats", "accuracy_curves.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def _plot_accuracy_cases(val_accuracies, test_accuracies, domain):
        """Plot worst case, average case and best case accuracies for TTA over multiple seeds"""
        # Validation Accuracy detailed plot
        plt.subplot(1, 2, 1)
        if isinstance(val_accuracies[0], (list, tuple, np.ndarray)):
            # Calculate min, mean, max across different runs/seeds
            val_accuracies = np.array(val_accuracies)
            val_min = np.min(val_accuracies, axis=0)
            val_mean = np.mean(val_accuracies, axis=0)
            val_max = np.max(val_accuracies, axis=0)
        
            plt.plot(val_mean, label='Average Case', color='green')
            plt.fill_between(range(len(val_mean)), val_min, val_max, 
                             color='green', alpha=0.2, label='Range (Min-Max)')
            plt.plot(val_min, '--', color='darkgreen', alpha=0.5, label='Worst Case')
            plt.plot(val_max, '--', color='lime', alpha=0.5, label='Best Case')
        else:
            plt.plot(val_accuracies, label='Validation Accuracy', color='green')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Domain {domain}: Validation Accuracy Analysis')
        plt.legend()
    
        # Test Accuracy detailed plot
        plt.subplot(1, 2, 2)
        if isinstance(test_accuracies[0], (list, tuple, np.ndarray)):
            # Calculate min, mean, max across different runs/seeds
            test_accuracies = np.array(test_accuracies)
            test_min = np.min(test_accuracies, axis=0)
            test_mean = np.mean(test_accuracies, axis=0)
            test_max = np.max(test_accuracies, axis=0)
        
            plt.plot(test_mean, label='Average Case', color='red')
            plt.fill_between(range(len(test_mean)), test_min, test_max, 
                             color='red', alpha=0.2, label='Range (Min-Max)')
            plt.plot(test_min, '--', color='darkred', alpha=0.5, label='Worst Case')
            plt.plot(test_max, '--', color='salmon', alpha=0.5, label='Best Case')
        else:
            plt.plot(test_accuracies, label='Test Accuracy', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Domain {domain}: Test Accuracy Analysis')
        plt.legend()


    def _plot_confusion_matrix(self, model, loader, domain_name, normalize=True):
        """Generates and plots confusion matrix compatible with DomainSubset"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            vmin, vmax = 0, 1
        else:
            fmt = 'd'
            vmin, vmax = None, None

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    vmin=vmin, vmax=vmax)
        plt.title(f'Domain {domain_name}: Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
       
        save_path = os.path.join(self.config['vis_dir'], "confusion_matrices", f'confusion_matrix_{domain_name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def _plot_global_confusion_matrix(self, domains, true_labels, pred_labels):
        """Plots a combined confusion matrix showing performance across all domains
    
        Args:
            domains: List of domain indices for each sample
            true_labels: List of true class labels
            pred_labels: List of predicted class labels
        """
        # Input Validation
        if len(domains) != len(true_labels) or len(true_labels) != len(pred_labels):
            raise ValueError("Input lists must have the same length")

        # Convert to numpy arrays
        domains = np.array(domains)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Create combined labels
        combined_true = [
            f"{self.domain_names[d]}_{self.class_names[l]}" 
            for d, l in zip(domains, true_labels)
        ]
    
        combined_pred = [
            f"{self.domain_names[d]}_{self.class_names[p]}" 
            for d, p in zip(domains, pred_labels)
        ]

        # Calculate confusion matrix
        unique_labels = sorted(set(combined_true + combined_pred))
        cm = confusion_matrix(combined_true, combined_pred, labels=unique_labels)

        # Normalize and plot
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)  # +epsilon to avoid division by zero
    
        plt.figure(figsize=(20, 18))
        sns.set(font_scale=0.7)
    
        # Smart label display
        show_every = max(1, len(unique_labels) // 20)
        display_labels = [label if i % show_every == 0 else '' 
                         for i, label in enumerate(unique_labels)]
    
        sns.heatmap(
            cm,
            xticklabels=display_labels,
            yticklabels=display_labels,
            cmap='Blues',
            annot=True,
            cbar_kws={'shrink': 0.8}
        )
    
        plt.title('Global Confusion Matrix Across All Domains', pad=20)
        plt.xlabel('Predicted (Domain_Class)')
        plt.ylabel('True (Domain_Class)')
    
        # Save
        save_path = os.path.join(self.config['vis_dir'], "confusion_matrices", 'global_confusion_matrix.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    

    def _plot_roc_pr_curves(self, model, loader, domain):
        """Plot ROC and Precision-Recall curves for each class"""
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, domains in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)                          #TODO 
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # One-vs-all for multi-class
        n_classes = len(self.class_names)
        plt.figure(figsize=(15, 6))
        
        # ROC curves
        plt.subplot(1, 2, 1)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Domain {domain}: ROC Curves')
        plt.legend(loc="lower right")
        
        # PR curves
        plt.subplot(1, 2, 2)
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve((all_labels == i).astype(int), all_probs[:, i])
            plt.plot(recall, precision, label=f'{self.class_names[i]}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Domain {domain}: Precision-Recall Curves')
        plt.legend(loc="lower left")
        
        plt.tight_layout()      #TODO self.config
        save_path = os.path.join(self.config['vis_dir'], "roc_pr", f'roc_pr_curves_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()
    

    def _visualize_embeddings_single_classes(self, model, loader, domain_name, n_samples=500):
        """Visualize feature embeddings with complete class representation"""
        model.eval()
        features = []
        labels = []
        domains = []
    
        # Remove last layer to get embeddings
        original_fc = model.fc
        model.fc = nn.Identity()
    
        # First pass: Collect all unique classes and domains
        all_classes = set()
        all_domains_in_data = set()
    
        with torch.no_grad():
            for inputs, lbls, doms in loader:
                all_classes.update(lbls.cpu().numpy())
                all_domains_in_data.update(doms.cpu().numpy())
    
        # Second pass: Collect features with balanced representation
        class_counts = {c: 0 for c in all_classes}
        samples_per_class = max(1, n_samples // len(all_classes))
    
        with torch.no_grad():
            for inputs, lbls, doms in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
            
                current_features = outputs.cpu().numpy()
                current_labels = lbls.cpu().numpy()
                current_domains = doms.cpu().numpy()
            
                # Select samples to maintain class balance
                for c in all_classes:
                    class_mask = (current_labels == c)
                    if np.sum(class_mask) > 0 and class_counts[c] < samples_per_class:
                        available_indices = np.where(class_mask)[0]
                        take_n = min(len(available_indices), 
                                    samples_per_class - class_counts[c])
                        selected_indices = np.random.choice(available_indices, take_n, replace=False)

                        features.extend(current_features[selected_indices])
                        labels.extend(current_labels[selected_indices])
                        domains.extend(current_domains[selected_indices])
                        class_counts[c] += take_n
            
                if len(features) >= n_samples:
                    break
    
        # Restore original model
        model.fc = original_fc
    
        features = np.array(features)[:n_samples]
        labels = np.array(labels)[:n_samples]
        domains = np.array(domains)[:n_samples]
    
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        tsne_results = tsne.fit_transform(features)
    
        plt.figure(figsize=(18, 8))
    
        # By class - ensure all classes are represented
        plt.subplot(1, 2, 1)
        for c in sorted(all_classes):
            mask = labels == c
            if np.any(mask):  # Only plot if samples exist
                plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                           label=self.class_names[c], alpha=0.6)
        plt.title(f'Domain {domain_name}: t-SNE by Class\n(Total samples: {len(labels)})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
        # By domain - ensure all domains are represented
        plt.subplot(1, 2, 2)
        for d in sorted(all_domains_in_data):
            mask = domains == d
            if np.any(mask):  # Only plot if samples exist
                plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                        label=self.domain_names[d], alpha=0.6)
        plt.title(f'Domain {domain_name}: t-SNE by Domain\n(Total samples: {len(labels)})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "tsne", f'tsne_{domain_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    
    def _visualize_embedded_dataset(self, model, loader, n_samples=500):
        """Visualize feature embeddings with complete class representation"""
        model.eval()
        features = []
        labels = []
        domains = []
    
        # Remove last layer to get embeddings
        original_fc = model.fc
        model.fc = nn.Identity()
    
        # First pass: Collect all unique classes and domains
        all_classes = set()
        all_domains_in_data = set()
    
        with torch.no_grad():
            for inputs, lbls, doms in loader:
                all_classes.update(lbls.cpu().numpy())
                all_domains_in_data.update(doms.cpu().numpy())
    
        # Second pass: Collect features with balanced representation
        class_counts = {c: 0 for c in all_classes}
        samples_per_class = max(1, n_samples // len(all_classes))
    
        with torch.no_grad():
            for inputs, lbls, doms in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
            
                current_features = outputs.cpu().numpy()
                current_labels = lbls.cpu().numpy()
                current_domains = doms.cpu().numpy()
            
                # Select samples to maintain class balance
                for c in all_classes:
                    class_mask = (current_labels == c)
                    if np.sum(class_mask) > 0 and class_counts[c] < samples_per_class:
                        available_indices = np.where(class_mask)[0]
                        take_n = min(len(available_indices), 
                                    samples_per_class - class_counts[c])
                        selected_indices = np.random.choice(available_indices, take_n, replace=False)

                        features.extend(current_features[selected_indices])
                        labels.extend(current_labels[selected_indices])
                        domains.extend(current_domains[selected_indices])
                        class_counts[c] += take_n
            
                if len(features) >= n_samples:
                    break
    
        # Restore original model
        model.fc = original_fc
    
        features = np.array(features)[:n_samples]
        labels = np.array(labels)[:n_samples]
        domains = np.array(domains)[:n_samples]
    
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        tsne_results = tsne.fit_transform(features)
    
        plt.figure(figsize=(18, 8))
    
        # By class - ensure all classes are represented
        plt.subplot(1, 2, 1)
        for c in sorted(all_classes):
            mask = labels == c
            if np.any(mask):
                plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                           label=self.class_names[c], alpha=0.6)
        plt.title(f'Embedded Dataset: t-SNE by Class\n(Total samples: {len(labels)})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "tsne", f'tsne_dataset.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    
    def _visualize_full_embedded_dataset(self, model, loader):
        """Visualize t-SNE of entire embedded dataset"""
        model.eval()
        features = []
        labels = []
        domains = []
    
        # Remove last layer to get embeddings
        original_fc = model.fc
        model.fc = nn.Identity()
    
        with torch.no_grad():
            for inputs, lbls, doms in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                features.append(outputs.cpu().numpy())
                labels.extend(lbls.cpu().numpy())
                domains.extend(doms.cpu().numpy())
    
        # Restore original model
        model.fc = original_fc
    
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)
        domains = np.array(domains)
    
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        tsne_results = tsne.fit_transform(features)
    
        plt.figure(figsize=(18, 8))

        unique_domains = np.unique(domains)
        unique_labels = np.unique(labels)

        domain_palette = sns.color_palette("bright", n_colors=len(unique_domains))
        class_palette = sns.color_palette("bright", n_colors=len(unique_labels))

        # Mapping: Domain-ID → Farbindex
        domain_to_color = {d: domain_palette[i] for i, d in enumerate(unique_domains)}
        class_to_color = {c: class_palette[i] for i, c in enumerate(unique_labels)}
    
        # === Plot 1: By class ===
        ax1 = plt.subplot(1, 2, 1)
        unique_classes = np.unique(labels)
        #palette_classes = sns.color_palette("tab10", len(unique_classes))
        for i, c in enumerate(unique_classes):
            mask = labels == c
            ax1.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                        label=self.class_names[c], alpha=0.6, color=class_to_color[c]) #color=palette_classes[i])
        ax1.set_title(f'Full Embedded Dataset: t-SNE by Class\n(Total samples: {len(labels)})')
        ax1.set_xlabel('t-SNE Dim 1')
        ax1.set_ylabel('t-SNE Dim 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # === Plot 2: By domain ===
        ax2 = plt.subplot(1, 2, 2)
        unique_domains = np.unique(domains)
        #palette_domains = sns.color_palette("Set2", len(unique_domains))
        for i, d in enumerate(unique_domains):
            mask = domains == d
            label = self.domain_names[d] if d < len(self.domain_names) else f"Domain {d}"
            ax2.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                        label=label, alpha=0.6, color=domain_to_color[d]) #color=palette_domains[i])
        ax2.set_title(f'Full Embedded Dataset: t-SNE by Domain\n(Total samples: {len(domains)})')
        ax2.set_xlabel('t-SNE Dim 1')
        ax2.set_ylabel('t-SNE Dim 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save plot
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "tsne", 'tsne_full_embedded_dataset.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        return tsne_results
    

    def _visualize_part_raw_dataset(self, loader, n_samples=500):
        """Visualize t-SNE of raw input data before classification: by class and by domain"""
        all_classes = set()
        all_domains = set()
    
        for batch in loader:
            _, label_batch, domain_batch = batch
            all_classes.update(label_batch.cpu().numpy())
            all_domains.update(domain_batch.cpu().numpy())
    
        # Calculate samples per class-domain combination
        n_combinations = len(all_classes) * len(all_domains)
        samples_per_combination = max(1, n_samples // n_combinations)
    
        # Second pass: Collect balanced samples
        images, labels, domains = [], [], []
        counts = {(c,d): 0 for c in all_classes for d in all_domains}
    
        for batch in loader:
            img_batch, label_batch, domain_batch = batch
            img_batch = img_batch.cpu().numpy()
            label_batch = label_batch.cpu().numpy()
            domain_batch = domain_batch.cpu().numpy()
        
            for img, label, domain in zip(img_batch, label_batch, domain_batch):
                if counts[(label, domain)] < samples_per_combination:
                    images.append(img)
                    labels.append(label)
                    domains.append(domain)
                    counts[(label, domain)] += 1
                
            if len(images) >= n_samples:
                break
    
        images = np.array(images)
        labels = np.array(labels)
        domains = np.array(domains)
    
        # Flatten images for t-SNE
        images_flat = images.reshape(len(images), -1)
    
        # Compute t-SNE with adjusted perplexity
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(images_flat)-1),
            max_iter=1000
        )
        embeddings = tsne.fit_transform(images_flat)
    
        # Create figure
        fig = plt.figure(figsize=(20, 8))

        unique_domains = np.unique(domains)
        unique_labels = np.unique(labels)

        domain_palette = sns.color_palette("bright", n_colors=len(unique_domains))
        class_palette = sns.color_palette("bright", n_colors=len(unique_labels))

        # Mapping: Domain-ID → Farbindex
        domain_to_color = {d: domain_palette[i] for i, d in enumerate(unique_domains)}
        class_to_color = {c: class_palette[i] for i, c in enumerate(unique_labels)}
    
        # Plot 1: By domain with all classes visible
        ax1 = plt.subplot(1, 2, 1)
        for domain_idx in np.unique(domains):
            mask = domains == domain_idx
            #color = plt.cm.tab10(domain_idx / len(self.domain_names))
            color = domain_to_color[domain_idx]
            label = self.domain_names[domain_idx] if domain_idx < len(self.domain_names) else f"Domain {domain_idx}"
            ax1.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=[color], label=label, alpha=0.6)

        ax1.set_title(f'Raw Data t-SNE by Domain\n(Total samples: {len(images)})')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
        # Plot 2: By class with all domains visible
        ax2 = plt.subplot(1, 2, 2)
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            #color = plt.cm.tab20(class_idx / len(self.class_names))
            color = class_to_color[class_idx]
            ax2.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=[color], label=self.class_names[class_idx], alpha=0.6)
    
        ax2.set_title(f'Raw Data t-SNE by Class\n(Total samples: {len(images)})')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
        # Save and close
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "tsne", 'selected_raw_data_tsne.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        return embeddings
    

    def _visualize_complete_raw_dataset(self, loader):
        """Visualize t-SNE of entire raw input data before classification by class and domain"""

        # Sammle alle Daten
        images, labels, domains = [], [], []
        for batch in loader:
            img_batch, label_batch, domain_batch = batch
            images.append(img_batch.cpu().numpy())
            labels.append(label_batch.cpu().numpy())
            domains.append(domain_batch.cpu().numpy())

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        domains = np.concatenate(domains, axis=0)

        # Flatten images for t-SNE
        images_flat = images.reshape(len(images), -1)

        # Compute t-SNE
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(images_flat) - 1),
            max_iter=1000
        )
        embeddings = tsne.fit_transform(images_flat)

        # Plotting
        fig = plt.figure(figsize=(20, 8))
        
        unique_domains = np.unique(domains)
        unique_labels = np.unique(labels)

        domain_palette = sns.color_palette("bright", n_colors=len(unique_domains))
        class_palette = sns.color_palette("bright", n_colors=len(unique_labels))

        # Mapping: Domain-ID → Farbindex
        domain_to_color = {d: domain_palette[i] for i, d in enumerate(unique_domains)}
        class_to_color = {c: class_palette[i] for i, c in enumerate(unique_labels)}

        ax1 = plt.subplot(1, 2, 1)
        for domain_idx in np.unique(domains):
            mask = domains == domain_idx
            color = domain_to_color[domain_idx]
            label = self.domain_names[domain_idx] if domain_idx < len(self.domain_names) else f"Domain {domain_idx}"
            ax1.scatter(embeddings[mask, 0], embeddings[mask, 1], c=[color], label=label, alpha=0.6)
        ax1.set_title(f'Raw Data t-SNE by Domain\n(Total samples: {len(images)})')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax2 = plt.subplot(1, 2, 2)
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            color = class_to_color[class_idx]
            ax2.scatter(embeddings[mask, 0], embeddings[mask, 1], c=[color], label=self.class_names[class_idx], alpha=0.6)
        ax2.set_title(f'Raw Data t-SNE by Class\n(Total samples: {len(images)})')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "tsne", 'complete_raw_data_tsne.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        return embeddings


    def visualize_resnet_tsne_blocks(
            self,
            model: nn.Module,
            loader: DataLoader,
            device: torch.device,
            block_names: List[str] = ['layer1', 'layer2', 'layer3', 'layer4'],
            n_samples: int = 1000,
            perplexity: int = 30,
            max_iter: int = 1000,
            random_state: int = 42
    ):
        """
        Generates publication-quality t-SNE plots showing domain vs class separation across ResNet blocks

        Args:
            model: ResNet model
            loader: DataLoader with inputs
            device: torch device (cpu/cuda)
            block_names: List of block names to visualize
            n_samples: Balanced samples per class/domain
            perplexity: t-SNE perplexity parameter
            max_iter: t-SNE iterations
            random_state: Random seed
        """
        model.eval()
    
        # Hook containers
        features = {name: [] for name in block_names}
        labels = []
        domains = []
    
        # Register hooks
        hooks = []
        for name, module in model.named_children():
            if name in block_names:
                def hook_factory(layer_name):
                    def hook(module, input, output):
                        # Global average pooling and flatten
                        pooled = nn.functional.adaptive_avg_pool2d(output, (1, 1))
                        features[layer_name].append(pooled.view(output.size(0), -1).cpu().detach())
                    return hook
                hooks.append(module.register_forward_hook(hook_factory(name)))
    
        # Collect all data first to ensure consistent sampling
        all_inputs = []
        all_labels = []
        all_domains = []
    
        with torch.no_grad():
            for inputs, lbls, doms in loader:
                all_inputs.append(inputs)
                all_labels.append(lbls.cpu().numpy())
                all_domains.append(doms.cpu().numpy())
    
        # Concatenate all collected data
        all_labels = np.concatenate(all_labels)
        all_domains = np.concatenate(all_domains)
    
        # Get unique classes and domains
        unique_classes = np.unique(all_labels)
        unique_domains = np.unique(all_domains)
    
        # Calculate balanced samples per group
        #samples_per_group = max(10, n_samples // (len(unique_classes) * len(unique_domains)))
        samples_per_group = max(20, n_samples // (len(unique_classes) * len(unique_domains)))
        selected_indices = []
        group_counts = DefaultDict(int)
    
        for idx, (l, d) in enumerate(zip(all_labels, all_domains)):
            if group_counts[(l, d)] < samples_per_group:
                selected_indices.append(idx)
                group_counts[(l, d)] += 1
            
            # Early stop when all groups have enough samples
            if len(group_counts) == len(unique_classes) * len(unique_domains) and \
            all(v >= samples_per_group for v in group_counts.values()):
                break
    
        # Process selected samples through model
        with torch.no_grad():
            for idx in selected_indices:
                """
                inputs = all_inputs[idx // loader.batch_size][idx % loader.batch_size].unsqueeze(0).to(device)
                _ = model(inputs)
                labels.append(all_labels[idx])
                domains.append(all_domains[idx])
                """
                batch_idx = idx // loader.batch_size
                item_idx = idx % loader.batch_size
                inputs = all_inputs[batch_idx][item_idx].unsqueeze(0).to(device)
                _ = model(inputs)
                labels.append(all_labels[idx])
                domains.append(all_domains[idx])
    
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
        # Convert to numpy arrays
        labels = np.array(labels)
        domains = np.array(domains)
    
        # Create output directory
        save_path = os.path.join(self.config['vis_dir'], "tsne")
    
        # Create combined figure for each layer
        for layer_name in block_names:
            if not features.get(layer_name) or len(features[layer_name]) == 0:
                continue
            
            layer_features = torch.cat(features[layer_name], dim=0).numpy()
        
            # Verify dimensions match
            if len(layer_features) != len(labels):
                print(f"Skipping {layer_name} due to dimension mismatch (features: {len(layer_features)}, labels: {len(labels)})")
                continue
        
            # Run t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=random_state
            )
            embeddings = tsne.fit_transform(layer_features)
        
            # Create figure with two subplots
            plt.figure(figsize=(16, 13))
        
            # Top: Domain visualization
            plt.subplot(2, 1, 1)
            for domain_idx in np.unique(domains):  # Verwende tatsächliche Domain-Indizes
                mask = domains == domain_idx
                if np.any(mask):
                    plt.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        label=self.domain_names[domain_idx],  # Korrekte Namenszuordnung
                        alpha=0.7,
                        s=120,
                        edgecolor='w',
                        linewidth=0.3
                    )
            plt.title(f'{layer_name}: Domain Separation', pad=20, fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.xticks([])
            plt.yticks([])
        
            # Bottom: Class visualization
            plt.subplot(2, 1, 2)
            for class_idx, class_name in enumerate(self.class_names):
                mask = labels == class_idx
                if np.any(mask):
                    plt.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        label=class_name,
                        alpha=0.7,
                        s=120,
                        edgecolor='w',
                        linewidth=0.3
                    )
            plt.title(f'{layer_name}: Class Separation', pad=20, fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.xticks([])
            plt.yticks([])
        
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_path, f'latent_space_{layer_name}.png'),
                bbox_inches='tight',
                dpi=300,
                transparent=False
            )
            plt.close()

    
    def _visualize_predictions(self, model, loader, domain_name, num_examples=5):
        """Visualize example predictions compatible with DomainSubset"""
        model.eval()
        fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples*3))

        if num_examples == 1:
            axes = axes.reshape(1, -1)
    
        examples_collected = 0
    
        for inputs, labels, _ in loader:
            if examples_collected >= num_examples:
                break
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        
            # Get predictions
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
        
            # Denormalize image
            img = inputs[0].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
        
            # Original image (left column)
            axes[examples_collected, 0].imshow(img)
            axes[examples_collected, 0].set_title(
                f'True: {self.class_names[labels[0].item()]}\n'
                f'Pred: {self.class_names[preds[0].item()]}'
            )
            axes[examples_collected, 0].axis('off')
        
            # Saliency map (right column)
            inputs.requires_grad_()
            outputs = model(inputs)
            loss = outputs[0, preds[0]]
            loss.backward()
        
            saliency = inputs.grad.data.abs().max(dim=1)[0][0].cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
            axes[examples_collected, 1].imshow(img)
            axes[examples_collected, 1].imshow(saliency, cmap='hot', alpha=0.5)
            axes[examples_collected, 1].set_title('Saliency Map')
            axes[examples_collected, 1].axis('off')
        
            examples_collected += 1
    
        plt.suptitle(f'Domain {domain_name}: Saliency Visualizations', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "heatmaps", f'saliency_{domain_name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    
    def _visualize_gradcam_predictions(self, model, loader, domain_name, num_examples=5, target_layer=None):
        """Visualize original images with Grad-CAM heatmaps"""
        model.eval()
    
        # If no target layer specified, find last conv layer automatically
        if target_layer is None:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
    
        # Hook containers
        feature_maps = []
        gradients = []
    
        def forward_hook(module, input, output):
            feature_maps.append(output.detach())
    
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
    
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
    
        # Create figure with 2 columns
        fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples*3))
        if num_examples == 1:
            axes = axes.reshape(1, -1)  # Ensure 2D array
    
        examples_collected = 0
    
        for inputs, labels, _ in loader:
            if examples_collected >= num_examples:
                break
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        
            # Clear previous hooks
            feature_maps.clear()
            gradients.clear()
        
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
            # Backward pass for Grad-CAM
            model.zero_grad()
            one_hot = torch.zeros_like(outputs)
            one_hot[0, preds[0]] = 1
            outputs.backward(gradient=one_hot, retain_graph=True)
        
            # Compute Grad-CAM
            if feature_maps and gradients:
                feature_map = feature_maps[0][0]
                gradient = gradients[0][0]
            
                # Pool gradients and weight feature maps
                weights = torch.mean(gradient, dim=[1, 2], keepdim=True)
                gradcam = torch.relu(torch.sum(weights * feature_map, dim=0))
            
                # Normalize
                gradcam = gradcam.cpu().numpy()
                gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
            
                # Denormalize original image
                img = inputs[0].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
            
                # Left column: Original image with labels
                axes[examples_collected, 0].imshow(img)
                axes[examples_collected, 0].set_title(
                    f'True: {self.class_names[labels[0].item()]}\n'
                    f'Pred: {self.class_names[preds[0].item()]}'
                )
                axes[examples_collected, 0].axis('off')
            
                # Right column: Grad-CAM overlay
                axes[examples_collected, 1].imshow(img)
                axes[examples_collected, 1].imshow(gradcam, cmap='jet', alpha=0.5)
                axes[examples_collected, 1].set_title('Grad-CAM Heatmap')
                axes[examples_collected, 1].axis('off')
            
                examples_collected += 1
    
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
    
        plt.suptitle(f'Domain {domain_name}: Grad-CAM Visualizations', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], "heatmaps", f'gradcam_clean_{domain_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    
    def _visualize_umap_embeddings(self, model, loader, domain_name, n_samples=1000):
        """Visualize dataset embeddings using UMAP"""
    
        model.eval()
        embeddings = []
        labels = []
        domains = []
    
        # Collect embeddings
        with torch.no_grad():
            for batch_idx, (inputs, batch_labels, batch_domains) in enumerate(loader):
                inputs = inputs.to(self.device)
                features = model._feature_extractor(inputs)  # Assuming your model has a .features method
                flattened_features = features.view(features.size(0), -1)
                embeddings.append(flattened_features.cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
                domains.append(batch_domains.cpu().numpy())
            
                if batch_idx * loader.batch_size >= n_samples:
                    break
    
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)
        domains = np.concatenate(domains)
    
        # Reduce dimensions with UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_embeddings = reducer.fit_transform(embeddings)
    
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': umap_embeddings[:, 0],
            'y': umap_embeddings[:, 1],
            'class': [self.class_names[l] for l in labels],
            'domain': [self.domain_names[d] for d in domains]
        })
    
        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='class',
            style='domain',
            palette='tab20',
            s=50,
            alpha=0.7
        )
    
        plt.title(f'UMAP Projection - {domain_name} Domain', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
        # Save
        save_path = os.path.join(self.config['vis_dir'], 'umap', f'umap_{domain_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
        # Optional: Return coordinates for further analysis
        return df
    

    def _visualize_raw_umap(self, loader, domain_name, n_samples=1000):
        """UMAP visualization of raw images (pixel space)"""
    
        pixels = []
        labels = []
        domains = []
    
        for batch_idx, (inputs, batch_labels, batch_domains) in enumerate(loader):
            # Flatten images
            flattened = inputs.view(inputs.size(0), -1).cpu().numpy()
            pixels.append(flattened)
            labels.append(batch_labels.cpu().numpy())
            domains.append(batch_domains.cpu().numpy())
        
            if batch_idx * loader.batch_size >= n_samples:
                break

        pixels = np.concatenate(pixels)
        labels = np.concatenate(labels)
        domains = np.concatenate(domains)
    
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_embeddings = reducer.fit_transform(pixels)
    
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': umap_embeddings[:, 0],
            'y': umap_embeddings[:, 1],
            'class': [self.class_names[l] for l in labels],
            'domain': [self.domain_names[d] for d in domains]
        })
    
        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='class',
            style='domain',
            palette='tab20',
            s=50,
            alpha=0.7
        )
    
        plt.title(f'UMAP Projection - {domain_name} Domain', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
        # Save
        save_path = os.path.join(self.config['vis_dir'], "umap", f'raw_umap_{domain_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
        # Optional: Return coordinates for further analysis
        return df


    def _plot_comparative_metrics(self, results):
        """Plot comparative metrics across domains using TrainingFramework results"""
        domain_names = []
        val_accs = []
        train_losses = []
        val_losses = []
        
        for domain_idx, domain_data in results['per_domain'].items():
            domain_names.append(domain_data['name'])
            val_accs.append(max(domain_data['epoch_val_accs']))
            train_losses.append(min(domain_data['epoch_train_losses']))
            val_losses.append(min(domain_data['epoch_val_losses']))
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Validation Accuracy
        ax1.bar(domain_names, val_accs)
        ax1.set_title('Best Validation Accuracy')
        ax1.set_ylim(0.7, 1)                    # y-axis limits for better visibility
        ax1.tick_params(axis='x', rotation=45)
        
        # Training Loss
        ax2.bar(domain_names, train_losses)
        ax2.set_title('Minimum Training Loss')
        ax2.tick_params(axis='x', rotation=45)
        
        # Validation Loss
        ax3.bar(domain_names, val_losses)
        ax3.set_title('Minimum Validation Loss')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(self.config['vis_dir'], 'comparative_metrics.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    
    def _evaluate_and_collect(self, model, loader):
        model.eval()
        preds, labels, domains = [], [], []
    
        with torch.no_grad():
            for inputs, lbls, doms in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, batch_preds = torch.max(outputs, 1)
            
                preds.extend(batch_preds.cpu().numpy())
                labels.extend(lbls.cpu().numpy())
                domains.extend(doms.cpu().numpy())
    
        return preds, labels, domains


    def _plot_all_domain_averages(
            run_data: Dict[str, List],
            metric_name: str,
            ylabel: str,
            title: str,
            save_path: str = None,
            show: bool = True
    ):
        """
        Plot the average curves for each domain in a single plot, plus an overall average curve.
        Handles runs with varying epoch lengths by extending shorter runs.

        :param run_data: Nested dictionary with run and domain data.
        :param metric_name: Name of the metric to plot.
        :param ylabel: Label for the y-axis.
        :param title: Title of the plot.
        :param save_path: Path to save the figure. Default is None.
        :param show: Whether to display the plot. Default is True.
        :return: None
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        max_epochs = max(
            max(len(run['epoch']) for run in runs)
            for runs in run_data.values()
        )
        all_epochs = np.arange(0, max_epochs)

        all_domain_means = []

        for domain, runs in run_data.items():
            extended_metrics = []

            for run in runs:
                run_epochs = np.array(run['epoch'])
                run_metrics = np.array(run[metric_name])

                if len(run_epochs) < max_epochs:
                    missing_epochs = np.arange(len(run_epochs), max_epochs)
                    run_epochs = np.concatenate([run_epochs, missing_epochs])
                    last_metric = run_metrics[-1]
                    run_metrics = np.concatenate([run_metrics, [last_metric] * len(missing_epochs)])

                extended_metrics.append(run_metrics)

            metrics = np.vstack(extended_metrics)

            mean_curve = np.mean(metrics, axis=0)
            all_domain_means.append(mean_curve)

            ax.plot(all_epochs, mean_curve, label=f"{domain} Average", alpha=0.7, linestyle="--")

        overall_mean_curve = np.mean(all_domain_means, axis=0)
        ax.plot(all_epochs, overall_mean_curve, label="Overall Average", color="black", linewidth=2, linestyle="-")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Average {title} Across Domains")
        ax.set_xlim(0, max_epochs - 1)
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=max_epochs - 1))

        if 'Accuracy' in title:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.3, alpha=0.8)
            ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.minorticks_on()
            spacing = 0.5 if 'Validation' in title else 1
            #ax.yaxis.set_major_locator(MultipleLocator(spacing))
        elif 'Loss' in title:
            ax.set_ylim(bottom=0)

        ax.legend(title="Domains & Overall Average", bbox_to_anchor=(1.05, 1), loc='upper left')

        if save_path:
            save_name = f'global_{metric_name}_plot.png'
            os.makedirs(f'{save_path}/plots', exist_ok=True)
            full_save_path = os.path.join(f'{save_path}/plots', save_name)
            plt.savefig(full_save_path, bbox_inches="tight")
            print(f"Plot saved at {full_save_path}")

        if show:
            plt.show()
        plt.close()

    
    """   Visualizations for TTA   """

    """
    def visualize_tta_tsne(
            self, 
            original_features: torch.Tensor, 
            augmented_features: Dict[str, torch.Tensor]
        ):
        all_features = [original_features.cpu().numpy()]
        labels = ["Original"]

        for domain, features in augmented_features.items():
            all_features.append(features.cpu().numpy())
            labels.append(f"Aug-{domain}")

        combined = np.concatenate(all_features, axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(combined)

        plt.figure(figsize=(12, 8))
        start_idx = 0
        for i, (feat, label) in enumerate(zip(all_features, labels)):
            end_idx = start_idx + len(feat)
            plt.scatter(embeddings[start_idx:end_idx, 0],
                        embeddings[start_idx:end_idx, 1],
                        label=label,
                        alpha=0.6)
            start_idx = end_idx

        plt.title("t-SNE: Original vs. TTA-Augmented Features")
        plt.legend()
        save_path = os.path.join(self.vis_dir, "tsne", "tta_tsne_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    """


    def visualize_tta_tsne(
            self, 
            original_features, 
            augmented_features, 
            raw_train_data, 
            test_domain
        ):
        """
        Visualisiert TSNE-Plots für TTA-Experimente
    
        Args:
            original_features: Features der originalen Testdomäne (vor Augmentation)
            augmented_features: Dict {target_domain: features} nach Augmentation
            train_features: Dict {domain: features} der Trainingsdomänen
        """
        try:     
            # Daten vorbereiten
            all_features, all_domains, all_point_types = [], [], []
            domain_colors = {
                'art_painting': "#0d73bc",
                'cartoon': '#ff7f0e', 
                'photo': "#16c316",
                'sketch': "#d82b2b"
                # für vlcs einfügen #TODO
            }
            augmented_color = '#9467bd'     # Lila für augmentierte Punkte
            test_domain_color = '#8c564b'
        
            # 1. Trainingsdomänen hinzufügen
            for domain, features in raw_train_data.items():
                all_features.append(features)
                all_domains.extend([domain] * len(features))
                all_point_types.extend(['train'] * len(features))
        
            # 2. Originale Testdomäne
            if isinstance(original_features, torch.Tensor):
                original_features = original_features.cpu().numpy()
            all_features.append(original_features)
            all_domains.extend([test_domain] * len(original_features))
            all_point_types.extend(['test'] * len(original_features))
        
            # 3. Augmentierte Versionen
            for target_domain, features in augmented_features.items():
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                all_features.append(features)
                all_domains.extend([test_domain] * len(features))
                all_point_types.extend(['augmented'] * len(features))
        
            # TSNE berechnen
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_results = tsne.fit_transform(np.concatenate(all_features))
        
            # Plot erstellen
            plt.figure(figsize=(14, 10))
        
            # Scatter plot für jeden Punkttyp
            for point_type in ['train', 'test', 'augmented']:
                mask = np.array(all_point_types) == point_type
                current_domains = np.array(all_domains)[mask]
            
                # Einmalige Farben für jede Domain
                colors = []
                for domain in current_domains:
                    if point_type == 'train':
                        colors.append(domain_colors[domain])
                    elif point_type == 'test':
                        colors.append(test_domain_color)
                    else:  # augmented
                        colors.append(augmented_color)
            
                # Marker setzen
                marker = 'o' if point_type in ['train', 'test'] else '^'
                label = {
                    'train': 'Training domains',
                    'test': f'Original {test_domain}',
                    'augmented': f'Augmented {test_domain}'
                }[point_type]
            
                plt.scatter(
                    tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=colors,
                    label=label,
                    alpha=0.7,
                    marker=marker,
                    s=60,
                    edgecolors='w',
                    linewidths=0.5
                )
        
            # 5. Legende erstellen
            legend_elements = []
            # Domänen in Legende
            for domain, color in domain_colors.items():
                legend_elements.append(Line2D([0], [0], 
                                           marker='o', 
                                           color='w', 
                                           label=domain,
                                           markerfacecolor=color, 
                                           markersize=10))
            # Original und Augmented in Legende
            legend_elements.append(Line2D([0], [0], 
                                   marker='o', 
                                   color='w', 
                                   label=f'Original {test_domain}',
                                   markerfacecolor=test_domain_color, 
                                   markersize=10))
            legend_elements.append(Line2D([0], [0], 
                                   marker='^', 
                                   color='w', 
                                   label=f'Augmented {test_domain}',
                                   markerfacecolor=augmented_color, 
                                   markersize=10))
        
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f"t-SNE: Feature Distribution (Test Domain: {test_domain})")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.tight_layout()
        
            # 6. Plot speichern
            os.makedirs(self.vis_dir, exist_ok=True)
            save_path = os.path.join(self.vis_dir, "tsne", f"tta_tsne_{test_domain}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        
        except Exception as e:
            print(f"Error in TTA TSNE visualization: {str(e)}")

    
    def visualize_tta_gradcam(
            self, 
            model,
            original_img: torch.Tensor,
            augmented_imgs: Dict[str, torch.Tensor]
    ):
        num_samples = len(augmented_imgs) + 1
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        # orig
        self._plot_single_gradcam(model, original_img, axes[0], "Original")

        # augmented
        for i, (domain, img) in enumerate(augmented_imgs.items(), start=1):
            self._plot_single_gradcam(model, img, axes[i], f"Aug-{domain}")

        plt.tight_layout()
        #plt.suptitle("Grad-CAM: Original vs. TTA-Augmentation", y=1.02)
        save_path = os.path.join(self.vis_dir, "heatmaps", "tta_gradcam.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


    def _plot_single_gradcam(self, model, img, axes, title):
        model.eval()

        if img is None:
            print("Warning: No image provided for Grad-CAM")
            return

        feature_maps, gradients = [], []

        target_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            print("Warning: No convolutional layer found for Grad-CAM")
            return

        def forward_hook(m, i, o):
            feature_maps.append(o.detach())
        def backward_hook(m, gi, go):
            gradients.append(go[0].detach())

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            img_tensor = img.clone().to(self.device)
            #img_tensor.requires_grad_(True)
            #img_tensor = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor

            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
                pass
            else:
                raise ValueError(f"Unexpected image shape: {img_tensor.shape}")

            with torch.enable_grad():   
                #img_tensor = img_tensor.to(self.device).unsqueeze(0)
                output = model(img_tensor)
                pred_class = output.argmax(dim=1)

                model.zero_grad()
                one_hot = torch.zeros_like(output)
                one_hot[0][pred_class] = 1
                output.backward(gradient=one_hot, retain_graph=True)

            if feature_maps and gradients:
                features = feature_maps[0][0]
                grads = gradients[0][0]

                weights = torch.mean(grads, dim=[1, 2], keepdim=True)
                cam = torch.relu((weights * features).sum(dim=0))

                cam = (cam - cam.min()) / (cam.max() - cam.min() - 1e-8)
                cam = cam.cpu().numpy()

            img_display = img_tensor[0].cpu().permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)

            axes[0].imshow(img_display)
            axes[0].set_title(f"{title}\nPred: {self.class_names[pred_class.item()]}")
            axes[0].axis('off')
           
            axes[1].imshow(img_display)
            axes[1].imshow(cam, cmap='jet', alpha=0.5)
            axes[1].set_title(f"{title} - Grad-Cam")
            axes[1].axis('off')

        except Exception as e:
            print(f"Error during Grad-CAM visualization: {str(e)}")
        
        finally:
            forward_handle.remove()
            backward_handle.remove()


    def plot_tta_confusion_matrices(
            self,
            original_results: Dict[str, np.ndarray],
            augmented_results: Dict[str, Dict[str, np.ndarray]],
            normalize: bool = True
    ):
        plt.figure(figsize=(15, 5 * (len(augmented_results) + 1)))

        plt.subplot(len(augmented_results) + 1, 1, 1)
        self._plot_single_confusion(
            preds=original_results['preds'],
            labels=original_results['labels'],
            title="Original Prediction",
            normalize=normalize
        )

        for i, (domain, results) in enumerate(augmented_results.items(), start=2):
            plt.subplot(len(augmented_results) + 1, 1, i)
            self._plot_single_confusion(
                preds=results['preds'],
                labels=results['labels'],
                title=f"TTA-Augmented: {domain}",
                normalize=normalize
            )
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, "confusion_matrices", "tta_confusion_comparison.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


    def _plot_single_confusion(self, preds: np.ndarray, labels: np.ndarray, 
                         title: str, normalize: bool):
        cm = confusion_matrix(labels, preds)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            vmin, vmax = 0, 1
        else:
            fmt = 'd'
            vmin, vmax = None, None
    
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    vmin=vmin, vmax=vmax)
    
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

    
    def plot_confidence_intervals(self, results: Dict[str, Dict]):
        plt.figure(figsize=(12, 6))

        domains = list(results['target_domains'].keys())
        means = [np.mean(results['target_domains'][d]['all_probs'], axis=0).mean() for d in domains]
        stds = [np.std(results['target_domains'][d]['all_probs'], axis=0).mean() for d in domains]

        plt.errorbar(domains, means, yerr=stds, fmt='o', capsize=5, label='Mean Confidence ± Std')

        plt.title("Prediction Confidence across Augmentation Domains")
        plt.ylabel("Mean Confidence Score")
        plt.xticks(rotation=45)
        plt.legend()

        save_path = os.path.join(self.vis_dir, "test_stats", "confidence_intervals.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def plot_feature_stats_heatmap(
            self,
            original_img: torch.Tensor,
            augmented_imgs: List[torch.Tensor]
    ):
        def get_stats(img):
            img = img.cpu().numpy()
            # Für (C, H, W) Tensor
            if img.ndim == 3:
                return {
                    'mean': img.mean(axis=(1, 2)),  # Mittelwert über Höhe und Breite
                    'std': img.std(axis=(1, 2))
                }
            # Für (B, C, H, W) Tensor
            elif img.ndim == 4:
                return {
                    'mean': img.mean(axis=(0, 2, 3)),  # Mittelwert über Batch, Höhe und Breite
                    'std': img.std(axis=(0, 2, 3))
                }
            else:
                raise ValueError(f"Unexpected image dimension: {img.ndim}")
        
        orig_stats = get_stats(original_img)
        aug_stats = [get_stats(img) for img in augmented_imgs]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        sns.heatmap([orig_stats['mean'], *[s['mean'] for s in aug_stats]],
                    ax=axes[0, 0], annot=True,
                    xticklabels=["R", "G", "B"],
                    yticklabels=["Original"] + [f"Aug {i}" for i in range(len(aug_stats))])
        axes[0, 0].set_title("Channel Means")

        sns.heatmap([orig_stats['std'], *[s['std'] for s in aug_stats]],
                    ax=axes[0, 1], annot=True,
                    xticklabels=["R", "G", "B"])
        axes[0, 1].set_title("Channel Stds")

        plt.suptitle("RGB Channel Statistics before/after Augmentation")
        save_path = os.path.join(self.vis_dir, "heatmaps", "rbg_stats_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def plot_prediction_consistency(
            self,
            original_probs: np.ndarray,
            augmented_probs: List[np.ndarray]
    ):
        plt.figure(figsize=(8, 8))

        for i, aug_probs in enumerate(augmented_probs):
            plt.scatter(original_probs.max(axis=1), aug_probs.max(axis=1),
                        alpha=0.3, label=f"Aug {i}")
            
        plt.plot([0, 1], [0, 1], 'k--', label="Perfect Consistency") 
        plt.xlabel("Original Confidence")
        plt.ylabel("Augmented Confidence")
        plt.title("Prediction Consistency: Original vs Augmented")
        plt.legend()

        save_path = os.path.join(self.vis_dir, "test_stats", "prediction_consistency.png")
        plt.savefig(save_path)
        plt.close()

    def plot_uncertainty_dropping(
            self,
            drop_grid,
            curve_uncert,
            curve_rand_mean,
            curve_rand_std,
            test_domain,
            mode):

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(drop_grid, curve_uncert, marker='o', linestyle='--', label='Uncertainty criteria')
        ax.plot(drop_grid, curve_rand_mean, linestyle='-.', label='Random criteria')
        lower = np.array(curve_rand_mean) - np.array(curve_rand_std)
        upper = np.array(curve_rand_mean) + np.array(curve_rand_std)
        ax.fill_between(drop_grid, lower, upper, alpha=0.15)
        ax.set_xlabel('Percentage of dropped cases')
        ax.set_ylabel('Accuracy')
        if test_domain == "LabelMe":
            ax.set_ylim(0.3, 1.0)
        elif test_domain == "SUN09" or test_domain == "sketch":
            ax.set_ylim(0.6, 1.0)
        elif test_domain == "Caltech101" or test_domain == "photo":
            ax.set_ylim(0.9, 1.0)
        else:
            ax.set_ylim(0.7, 1.0)
        ax.set_title(f'{test_domain} - {mode}')
        ax.legend(loc='lower left')
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, "training_curves", f'perf_vs_drop_{test_domain}_{mode}.png')
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    
    def plot_calibration_vs_variance(
        self,
        per_sample_uncertainty,
        per_sample_correct,
        test_domain: str = None,
        mode: str = None,
        nbins: int = 10,
        save_dir: str = None,
        filename: str = None,
        show_scatter: bool = False,
        scatter_frac: float = 0.1,
        random_state: int = 0,
        ):
        u = np.asarray(per_sample_uncertainty, dtype=np.float64)
        c = np.asarray(per_sample_correct, dtype=np.float64)

        if u.ndim != 1 or c.ndim != 1 or u.size != c.size:
            raise ValueError("per_sample_uncertainty und per_sample_correct have to be 1D and same length.")

        valid = np.isfinite(u) & np.isfinite(c)
        u, c = u[valid], c[valid]
        if u.size == 0:
            raise ValueError("No valid data (NaN/inf).")

        umax = np.max(u)
        if umax <= 0:
            umax = 1e-12

        bin_edges = np.linspace(0.0, umax, nbins + 1)
        bin_idx = np.digitize(u, bin_edges, right=True)

        acc_per_bin = np.full(nbins, np.nan, dtype=np.float64)
        n_per_bin   = np.zeros(nbins, dtype=np.int64)
        ci95_hw     = np.full(nbins, np.nan, dtype=np.float64)

        for i in range(1, nbins + 1):
            mask = (bin_idx == i)
            n = int(np.sum(mask))
            n_per_bin[i - 1] = n
            if n > 0:
                p = float(np.mean(c[mask]))
                acc_per_bin[i - 1] = p
                se = np.sqrt(max(p * (1.0 - p), 0.0) / n)
                ci95_hw[i - 1] = 1.96 * se

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        m = np.isfinite(acc_per_bin)
        ax.errorbar(
            bin_centers[m],
            acc_per_bin[m],
            yerr=ci95_hw[m],
            fmt='o-',
            capsize=3,
            linewidth=1.5,
            markersize=4,
        )

        if show_scatter:
            rng = np.random.default_rng(random_state)
            n_all = u.size
            k = int(np.clip(np.round(scatter_frac * n_all), 0, n_all))
            if k > 0:
                idx = rng.choice(n_all, size=k, replace=False)
                ax.scatter(u[idx], c[idx], s=8, alpha=0.2)

        title_parts = ["Accuracy vs. per-sample variance"]
        if test_domain: title_parts.append(f"Test: {test_domain}")
        if mode:        title_parts.append(f"Mode: {mode}")
        ax.set_title(" | ".join(title_parts))
        ax.set_xlabel("Per-sample variance (uncertainty)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', linewidth=0.8)

        if save_dir is None:
            save_dir = getattr(self, "vis_dir", None) or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            suffix = []
            if test_domain: suffix.append(test_domain)
            if mode:        suffix.append(mode)
            base = "reliability_vs_variance" + ("_" + "_".join(suffix) if suffix else "")
            filename = base + ".png"

        fig_path = os.path.join(save_dir, filename)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        csv_path = os.path.join(save_dir, os.path.splitext(filename)[0] + "_binned.csv")
        try:
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["bin_left", "bin_right", "bin_center", "n", "acc", "ci95_halfwidth"])
                for i in range(nbins):
                    writer.writerow([
                        f"{bin_edges[i]:.10g}",
                        f"{bin_edges[i+1]:.10g}",
                        f"{bin_centers[i]:.10g}",
                        int(n_per_bin[i]),
                        "" if not np.isfinite(acc_per_bin[i]) else f"{acc_per_bin[i]:.10g}",
                        "" if not np.isfinite(ci95_hw[i]) else f"{ci95_hw[i]:.10g}",
                    ])
        except Exception:
            pass