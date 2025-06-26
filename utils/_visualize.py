import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import umap.umap_ as umap
from typing import List, Dict, DefaultDict
from data._datasets import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import torch
from data._datasets import DOMAIN_NAMES


class Visualizer:
    def __init__(self, config, class_names, domain_names, vis_dir): #, dataset):
        self.config = config
        self.class_names = class_names
        self.domain_names = domain_names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vis_dir = self.config['vis_dir']

        subdirs = [
            "training_curves",
            "confusion_matrices", 
            "roc_pr",
            "heatmaps",
            "tsne",
            "umap"
        ]

        for subdir in subdirs:
            dir_path = os.path.join(self.vis_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)


    def _plot_training_curves(self, train_losses, val_losses, val_accuracies, test_accuracies, domain):
        """Plot training and validation metrics over epochs"""
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Domain {domain}: Training vs Validation Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')

        if test_accuracies is not None:
            plt.plot(test_accuracies, label='Test Accuracy', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Domain {domain}: Validation and Test Accuracy')
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, "training_curves", f'training_curves_domain_{domain}.png')
        plt.savefig(save_path)
        plt.close()
    

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
       
        save_path = os.path.join(self.vis_dir, "confusion_matrices", f'confusion_matrix_{domain_name}.png')
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
        save_path = os.path.join(self.vis_dir, "confusion_matrices", 'global_confusion_matrix.png')
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
        save_path = os.path.join(self.vis_dir, "roc_pr", f'roc_pr_curves_domain_{domain}.png')
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
        save_path = os.path.join(self.vis_dir, "tsne", f'tsne_{domain_name}.png')
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
        save_path = os.path.join(self.vis_dir, "tsne", f'tsne_dataset.png')
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
        save_path = os.path.join(self.vis_dir, "tsne", 'tsne_full_embedded_dataset.png')
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
        save_path = os.path.join(self.vis_dir, "tsne", 'selected_raw_data_tsne.png')
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
        save_path = os.path.join(self.vis_dir, "tsne", 'complete_raw_data_tsne.png')
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
        save_path = os.path.join(self.vis_dir, "tsne")
    
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
        save_path = os.path.join(self.vis_dir, "heatmaps", f'saliency_{domain_name}.png')
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
        save_path = os.path.join(self.vis_dir, "heatmaps", f'gradcam_clean_{domain_name}.png')
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
        save_path = os.path.join(self.vis_dir, 'umap', f'umap_{domain_name}.png')
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
        save_path = os.path.join(self.vis_dir, "umap", f'raw_umap_{domain_name}.png')
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
        save_path = os.path.join(self.vis_dir, 'comparative_metrics.png')
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