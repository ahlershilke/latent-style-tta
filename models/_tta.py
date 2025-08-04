import os
import torch
import numpy as np
import random
import json
import argparse
#from sklearn.metrics import accuracy_score
import torch.nn as nn
import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Any
from torch.utils.data import DataLoader
import logging
from torchmetrics import Accuracy
from collections import defaultdict
from ._styleextraction import StyleStatistics
from ._resnet import resnet50
from data._datasets import get_dataset, DOMAIN_NAMES
from utils._visualize import Visualizer


class AggregationStrategy(Enum):
    AVERAGE = auto()
    MAJORITY_VOTE = auto()
    MAX_CONFIDENCE = auto()
    WEIGHTED_AVERAGE = auto()


class FeatureAggregationStrategy(Enum):
    MEAN = auto()
    MAX = auto()
    CONCAT = auto()


class SeedManager:
    def __init__(self, base_seed=42):
        self.base_seed = base_seed
        
    def set_seed(self, seed=None):
        """Set all relevant random seeds for reproducibility"""
        seed = self.base_seed if seed is None else seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class TTAClassifier(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 stats_root_path: Optional[str] = None, 
                 test_domain: str = None, 
                 device: str = 'cuda', 
                 num_classes: Optional[int] = None,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.AVERAGE,
                 feature_aggregation: FeatureAggregationStrategy = FeatureAggregationStrategy.MEAN,
                 verbose: bool = False,
                 seed: Optional[int] = None,
                 class_names: Optional[List[str]] = None,
                 domain_names: Optional[List[str]] = None,
                 mode: str = None,
                 vis_dir: Optional[str] = None
                 ):
        """
        TTA Classifier.
        
        Args:
            model: Pre-trained model with feature extractor and classifier
            stats_path: Path to .pth file containing μ and σ stats
            device: Device to run computations on
            num_classes: Optional number of classes for new classifier
            aggregation_strategy: How to aggregate predictions from different augmentations
            feature_aggregation: How to aggregate features from different layers
            verbose: Whether to print debug information
        """
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.aggregation_strategy = aggregation_strategy
        self.feature_aggregation = feature_aggregation
        self.alpha = 0.5
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)
        self.mode = mode
        self.test_domain = test_domain
        self.stats_root_path = stats_root_path
        
        self.seed = seed if seed is not None else 42
        self.seed_manager = SeedManager(self.seed)
        self.seed_manager.set_seed()


        #self.seed_manager = SeedManager(seed if seed is not None else 42)
        #self.seed_manager.set_seed()
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing TTAClassifier with verbose logging")

        # freeze model and setup
        self.model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        
        # Ensure model is in eval mode (important for BatchNorm/Dropout)
        self.model.eval()
        
        self.feature_extractor = model.to(device)
        
        # Add linear classifier if needed
        if num_classes is not None:
            feature_dim = self._get_feature_dim()
            self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        else:
            self.classifier = None
            
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)

        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(num_classes)]
        self.domain_names = domain_names #if domain_names is not None else [f"domain_{i}" for i in range(4)]

        self.style_stats = self._load_style_stats()

        """
        self.style_stats = StyleStatistics(
            num_domains=len(model.style_stats.domain_names),  # Anzahl Domänen
            num_layers=model.style_stats.num_layers,
            domain_names=self.domain_names,
            mode="average"#,     Anzahl Layer
            #mode="average"                                    # oder "selective"
        )
        """

        self.visualizer = Visualizer(
            class_names=self.class_names,
            domain_names=self.domain_names,
            config={},
            vis_dir=vis_dir
        )
        
        #if stats_path:
         #   self.style_stats.load_state_dict(torch.load(stats_path))
        
        # Hook for feature extraction
        self.features = {}
        self._register_hooks()
        
        # For uncertainty estimation
        self.entropy = nn.CrossEntropyLoss(reduction='none')
        
        # For weighted average aggregation
        self.domain_weights = None
        if self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            self._init_domain_weights()


        """
        self.stats_path = os.path.join(
            self.stats_root_path,
            f"seed_{self.seed}",
            "style_stats",
            f"test_{self.test_domain}",
            self.modus,
            f"style_stats_{self.test_domain}_{self.modus}.pth"
        )

        if os.path.exists(self.stats_path):
            self.style_stats.load_state_dict(torch.load(self.stats_path))
        elif verbose:
            print(f"Warning: Stats file not found at {self.stats_path}")
        """


    def _load_style_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load style statistics for all training domains"""
        stats_dict = {}
        train_domains = [d for d in self.domain_names if d != self.test_domain]
        
        for domain in train_domains:
            stats_path = os.path.join(
                self.stats_root_path,
                f"seed_{self.seed}",
                "style_stats",
                f"test_{self.test_domain}",
                self.mode,
                f"style_stats_{domain}_{self.mode}.pth"
            )
            
            if os.path.exists(stats_path):
                stats_dict[domain] = torch.load(stats_path, map_location=self.device)
            elif self.verbose:
                print(f"Warning: No stats found for domain {domain} at {stats_path}")
        
        return stats_dict


    def _load_stats(self, stats_path: str):
        """Load statistics with validation"""
        stats = torch.load(stats_path)
        self.domain_stats = stats.get('domain', {})
        self.layer_stats = stats.get('layer', {})
        
        if self.verbose:
            self.logger.info(f"Loaded stats with {len(self.domain_stats)} domains and {len(self.layer_stats)} layers")
            for layer, domains in self.layer_stats.items():
                self.logger.info(f"Layer {layer} has stats for domains: {list(domains.keys())}")

    
    def _load_stats_for_domain(self, target_domain: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load statistics for a specific target domain"""
        stats_path = os.path.join(self.stats_root_path,
                                  f"seed_{self.seed}",
                                  "style_stats",
                                  f"test_{self.test_domain}",
                                  self.mode,
                                  f"style_stats{target_domain}_{self.mode}.pth")

        if not os.path.exists(stats_path):
            if self.verbose:
                print(f"Warning: Stats file not found at {stats_path}")
            return None
        
        stats_dict = torch.load(stats_path, map_location=self.device)

        if target_domain not in stats_dict:
            if self.verbose:
                print(f"Warning: No stats found for target domain {target_domain}")
            return None
        
        return stats_dict[target_domain]


    def _init_domain_weights(self):
        """Initialize domain weights based on their frequency in stats"""
        domain_counts = defaultdict(int)
        for layer_stats in self.layer_stats.values():
            for domain in layer_stats.keys():
                domain_counts[domain] += 1
        
        total = sum(domain_counts.values())
        self.domain_weights = {domain: count/total for domain, count in domain_counts.items()}
        
        if self.verbose:
            self.logger.info(f"Initialized domain weights: {self.domain_weights}")


    def _register_hooks(self):
        """Register hooks to capture features at specific layers"""
        self.features = {}
        self.hooks = []

        """
        for layer_name in self.layer_stats.keys():
            try:
                layer = dict([*self.feature_extractor.named_modules()])[layer_name]
                
                def hook_fn(module, input, output, name=layer_name):
                    self.features[name] = output
                
                layer.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, layer_name))
                
                if self.verbose:
                    self.logger.info(f"Registered hook for layer: {layer_name}")
            except KeyError:
                if self.verbose:
                    self.logger.warning(f"Layer {layer_name} not found in model")
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                def hook_factory(n):
                    def hook(m, i, o):
                        self.features[n] = o.detach()
                    return hook
                self.hooks.append(module.register_forward_hook(hook_factory(name)))


    def _get_feature_dim(self) -> int:
        """Get feature dimension by forward pass"""
        test_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(test_input)
        return features.shape[1]


    def _aggregate_features(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate features from multiple layers according to strategy.
        
        Args:
            features_dict: Dictionary of {layer_name: features}
            
        Returns:
            Aggregated features
        """
        features = list(features_dict.values())
        
        if self.feature_aggregation == FeatureAggregationStrategy.MEAN:
            return torch.mean(torch.stack(features), dim=0)
        elif self.feature_aggregation == FeatureAggregationStrategy.MAX:
            return torch.max(torch.stack(features), dim=0)[0]
        elif self.feature_aggregation == FeatureAggregationStrategy.CONCAT:
            return torch.cat(features, dim=1)
        elif self.feature_aggregation == FeatureAggregationStrategy.ATTENTION:
            # Simple attention mechanism (could be enhanced)
            weights = torch.softmax(torch.randn(len(features)), dim=0).to(self.device)
            return sum(w * f for w, f in zip(weights, features))
        else:
            raise ValueError(f"Unknown feature aggregation strategy: {self.feature_aggregation}")


    def _augment_features(self, 
                          features: torch.Tensor, 
                          layer_idx: int, 
                          target_domain: str) -> torch.Tensor:
        """
        Apply cross-domain feature augmentation using pre-saved style statistics.
    
        Args:
            features: Input features to augment (shape: [B, C, H, W])
            layer_idx: Layer index to identify which statistics to use
            target_domain: Target domain index for cross-domain mixing
        
        Returns:
            Augmented features with cross-domain mixed statistics
        """
        if target_domain not in self.style_stats:
            return features
        
        #neu
        domain_stats = self.style_stats[target_domain]

        B = features.size(0)
        device = features.device
    
        # 1. Compute current features statistics
        mu = features.mean(dim=[2, 3], keepdim=True)
        var = features.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-8).sqrt()
        mu, sig = mu.detach(), sig.detach()
    
        # 2. Normalize features
        features_normed = (features - mu) / sig

        """
        # 3. Load pre-saved statistics for target domain
        domain_stats = self._load_stats_for_domain(target_domain)
        if domain_stats is None:
            return features
        """
        
        if self.mode.startswith('single'):
            target_mu = domain_stats['mu'].to(device)
            target_sig = domain_stats['sig'].to(device)
        elif self.mode.startswith('selective'):
            layer_key = f'layer_{layer_idx}_mu'
            if layer_key not in domain_stats:
                if self.verbose:
                    print(f"Warning: No stats found for layer {layer_idx} in selective mode")
                    return features
            target_mu = domain_stats[layer_key].to(device)
            target_sig = domain_stats[f'layer_{layer_idx}_sig'].to(device)
        elif self.mode == "average":
            target_mu = domain_stats['mu'].to(device)
            target_sig = domain_stats['sig'].to(device)
        else:
            if self.verbose:
                print(f"Warning: Unknown mode {self.mode}")
            return features
    
        if target_mu.dim() == 1:
            target_mu = target_mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            target_sig = target_sig.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
        # Ensure we have enough samples (repeat if necessary)
        if target_mu.size(0) < B:
            repeat_times = (B // target_mu.size(0)) + 1
            target_mu = target_mu.repeat(repeat_times, 1, 1, 1)[:B]
            target_sig = target_sig.repeat(repeat_times, 1, 1, 1)[:B]
        
        half = B // 2

        perm_a = torch.randperm(half)
        perm_b = torch.randperm(B - half)

        perm = torch.cat([
            torch.arange(half, B)[perm_b],
            torch.arange(0, half)[perm_a]
        ], dim=0)
        
        target_mu = target_mu[perm]
        target_sig = target_sig[perm]
    
        lam = self.beta.sample((B, 1, 1, 1)).to(device)
    
        mu_mix = mu * lam + target_mu * (1 - lam)
        sig_mix = sig * lam + target_sig * (1 - lam)
    
        # denormalize with mixed statistics
        augmented_features = features_normed * sig_mix + mu_mix
    
        return augmented_features
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass"""
        features = self.feature_extractor(x)
        if self.classifier:
            return self.classifier(features)
        return features


    def _aggregate_predictions(self, 
                             all_probs: List[torch.Tensor],
                             domain: Optional[str] = None) -> torch.Tensor:
        """
        Aggregate predictions according to the configured strategy.
        
        Args:
            all_probs: List of probability tensors from different augmentations
            domain: Current domain (for weighted average)
            
        Returns:
            Aggregated probabilities
        """
        if self.aggregation_strategy == AggregationStrategy.AVERAGE:
            return torch.mean(torch.stack(all_probs), dim=0)
        elif self.aggregation_strategy == AggregationStrategy.MAJORITY_VOTE:
            preds = torch.stack([torch.argmax(p, dim=1) for p in all_probs])
            votes = torch.mode(preds, dim=0)[0]
            return torch.zeros_like(all_probs[0]).scatter_(1, votes.unsqueeze(1), 1)
        elif self.aggregation_strategy == AggregationStrategy.MAX_CONFIDENCE:
            confidences = torch.stack([torch.max(p, dim=1)[0] for p in all_probs])
            max_idx = torch.argmax(confidences, dim=0)
            return torch.stack([all_probs[i][j] for j, i in enumerate(max_idx)])
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            weight = self.domain_weights.get(domain, 1.0) if domain else 1.0
            weighted = [p * weight for p in all_probs]
            return torch.sum(torch.stack(weighted), dim=0) / sum(weight for _ in all_probs)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")

    """
    def predict(self, 
               dataloader: torch.utils.data.DataLoader, 
               num_augments: int = 3,
               return_uncertainty: bool = False) -> dict:
        
        Perform feature-level TTA prediction with enhanced functionality.
        
        Args:
            dataloader: DataLoader with test data
            num_augments: Number of augmentations per sample
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            dict: Dictionary containing:
            - 'accuracy': Overall accuracy
            - 'all_probs': All class probabilities
            - 'variance': Variance of softmax probabilities across augmentations
            - 'disagreement': Prediction disagreement (variation ratio)
            - 'per_sample_metrics': Dictionary with metrics for each sample
        
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        uncertainties = []

        all_augmentation_probs = []  # To store probs from each augmentation
        per_sample_metrics = {
            'variance': [],
            'disagreement': []
        }
        
        available_domains = [d for d in self.domain_names if d != self.test_domain]
        
        if self.verbose:
            self.logger.info(f"Starting prediction with available domains: {available_domains}")
            self.logger.info(f"Using aggregation strategy: {self.aggregation_strategy}")
            self.logger.info(f"Using feature aggregation: {self.feature_aggregation}")

        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                labels = batch[1] if len(batch) > 1 else None
            
                batch_probs = []
                batch_logits = []
                augmentation_probs = []

                # Original (non-augmented)
                logits = self(images)
                probs = self.softmax(logits)
                batch_probs.append(probs)
                batch_logits.append(logits)
                augmentation_probs.append(probs.cpu().numpy())

                # Get augmented predictions for each domain
                for domain in available_domains:
                    # Forward to get features
                    _ = self.feature_extractor(images)

                    augmented_features = {}
                    for layer_name, orig_features in self.features.items():
                        augmented = self._augment_features(orig_features, layer_name, target_domain=domain)
                        augmented_features[layer_name] = augmented

                    # Aggregate features from multiple layers
                    aggregated_features = self._aggregate_features(augmented_features)

                    if hasattr(self.feature_extractor, 'forward_with_features'):
                        features = self.feature_extractor.forward_with_features(images, augmented_features)
                    else:
                        # Fallback - use aggregated features directly
                        features = aggregated_features

                    logits = self.classifier(features) if self.classifier else features
                    probs = self.softmax(logits)
                    
                    batch_probs.append(probs)
                    batch_logits.append(logits)
                    augmentation_probs.append(probs.cpu().numpy())

                all_augmentation_probs.append(np.stack(augmentation_probs))
            
                # Calculate variance across augmentations for each sample
                batch_variance = np.var(np.stack(augmentation_probs), axis=0).mean(axis=1)  # Mean over classes
                per_sample_metrics['variance'].append(batch_variance)
            
                # Calculate prediction disagreement (variation ratio)
                batch_preds = np.argmax(np.stack(augmentation_probs), axis=2)  # Shape: (augmentations, batch_size)
                mode_counts = np.array([np.bincount(batch_preds[:,i]).max() for i in range(batch_preds.shape[1])])
                batch_disagreement = 1 - mode_counts / len(augmentation_probs)
                per_sample_metrics['disagreement'].append(batch_disagreement)

                # Aggregate predictions according to strategy
                avg_probs = self._aggregate_predictions(batch_probs)
                preds = torch.argmax(avg_probs, dim=1)

                all_probs.append(avg_probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

                if labels is not None:
                    all_labels.append(labels.numpy())
                    # Update torchmetrics accuracy
                    self.accuracy.update(preds, torch.tensor(labels, device=self.device))

        results = {}
    
        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels)
            results['accuracy'] = self.accuracy.compute().item()
    
        results['all_probs'] = np.concatenate(all_probs, axis=0)

        # Calculate overall variance and disagreement metrics
        all_augmentation_probs = np.concatenate(all_augmentation_probs, axis=1)  # Shape: (augmentations, num_samples, num_classes)
        results['variance'] = np.mean(np.var(all_augmentation_probs, axis=0), axis=1)  # Per-sample variance
    
        all_preds_aug = np.argmax(all_augmentation_probs, axis=2)  # Shape: (augmentations, num_samples)
        mode_counts = np.array([np.bincount(all_preds_aug[:,i]).max() for i in range(all_preds_aug.shape[1])])
        results['disagreement'] = 1 - mode_counts / all_augmentation_probs.shape[0]
    
        # Also include per-sample metrics
        results['per_sample_metrics'] = {
            'variance': np.concatenate(per_sample_metrics['variance']),
            'disagreement': np.concatenate(per_sample_metrics['disagreement'])
        }
    
        return results
        """
    

    def predict(self, dataloader: torch.utils.data.DataLoader, num_augments: int = 3, return_uncertainty: bool = False) -> dict:
        """
        Perform feature-level TTA prediction with enhanced functionality.
    
        Args:
            dataloader: DataLoader with test data
            num_augments: Number of augmentations per sample
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            dict: Dictionary containing:
            - 'accuracy': Overall accuracy
            - 'all_probs': All class probabilities
            - 'variance': Variance of softmax probabilities across augmentations
            - 'disagreement': Prediction disagreement (variation ratio)
            - 'per_sample_metrics': Dictionary with metrics for each sample
            - 'test_domain': Current test domain name
            - 'augmentation_domains': List of domains used for augmentation
            - 'mode': Style statistics mode used
        """

        #test_loader = ...  # Ihr DataLoader
        #first_batch = next(iter(test_loader))
        #print(f"Number of elements in batch: {len(first_batch)}")
        #print(f"Shapes/types: {[type(x) for x in first_batch]}")

        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        all_augmentation_probs = []
        per_sample_metrics = {'variance': [], 'disagreement': []}
    
        available_domains = [d for d in self.domain_names if d != self.test_domain]
    
        # Register hooks for feature extraction
        """features = {}
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                def hook_factory(n):
                    def hook(m, i, o):
                        features[n] = o.detach()
                    return hook
                hooks.append(module.register_forward_hook(hook_factory(name)))
        """
                
        try:
            with torch.no_grad():
                for batch in dataloader:
                    print(f"Number of elements in batch: {len(batch)}")
                    print(f"Shapes/types: {[type(x) for x in batch]}")
                    images = batch[0].to(self.device)
                    labels = batch[1] if len(batch) > 1 else None
                    _ = batch[2] if len(batch) > 2 else None
                    batch_probs = []
                    augmentation_probs = []

                    # Original prediction
                    _ = self.model(images)  # Populates features dict
                    #orig_logits = self.model.classifier(features[list(features.keys())[-1]])
                    #orig_probs = self.softmax(orig_logits)
                    #batch_probs.append(orig_probs)
                    #augmentation_probs.append(orig_probs.cpu().numpy())
                    layer_features = [
                        self.model._feature_maps[f"Layer{i+1}"]
                        for i in range(4)
                        if f"layer{i+1}" in self.model._feature_maps
                    ]

                    # Augmented predictions
                    for domain in available_domains:
                        try:
                            augmented_features = []
                            #for layer_name, feat in features.items():
                            for layer_name, feat in enumerate(layer_features):
                                augmented = self._augment_features(feat, layer_name, domain)
                                #augmented_features[layer_name] = augmented
                                augmented_features.append(augmented)

                            # Use last layer features for prediction
                            #aug_logits = self.model.classifier(augmented_features[list(augmented_features.keys())[-1]])
                            combined = torch.mean(torch.stack(augmented_features), dim=0)
                            aug_logits = self.model.classifier(combined)
                            aug_probs = self.softmax(aug_logits)
                        
                            batch_probs.append(aug_probs)
                            augmentation_probs.append(aug_probs.cpu().numpy())
                    
                        except Exception as e:
                            if self.verbose:
                                print(f"Skipping domain {domain} due to error: {str(e)}")
                            continue

                    # Calculate metrics
                    all_augmentation_probs.append(np.stack(augmentation_probs))
                    batch_variance = np.var(np.stack(augmentation_probs), axis=0).mean(axis=1)
                    per_sample_metrics['variance'].append(batch_variance)
                
                    batch_preds = np.argmax(np.stack(augmentation_probs), axis=2)
                    mode_counts = np.array([np.bincount(batch_preds[:,i]).max() for i in range(batch_preds.shape[1])])
                    batch_disagreement = 1 - mode_counts / len(augmentation_probs)
                    per_sample_metrics['disagreement'].append(batch_disagreement)

                    # Aggregate predictions
                    avg_probs = self._aggregate_predictions(batch_probs)
                    preds = torch.argmax(avg_probs, dim=1)
                
                    all_probs.append(avg_probs.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                
                    if labels is not None:
                        all_labels.append(labels.numpy())
                        self.accuracy.update(preds, torch.tensor(labels, device=self.device))

        finally:
            # Remove hooks
            #for h in hooks:
             #   h.remove()
             print("done")

        # Prepare results
        results = {
            'accuracy': self.accuracy.compute().item() if len(all_labels) > 0 else None,
            'all_probs': np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else None,
            'test_domain': self.test_domain,
            'augmentation_domains': available_domains,
            'mode': self.mode,
            'seed': self.seed
        }

        if return_uncertainty:
            all_aug_probs = np.concatenate(all_augmentation_probs, axis=1) if len(all_augmentation_probs) > 0 else None
            results.update({
                'variance': np.mean(np.var(all_aug_probs, axis=0), axis=1) if all_aug_probs is not None else None,
                'disagreement': 1 - (np.array([np.bincount(all_aug_probs[:,i]).max() 
                                             for i in range(all_aug_probs.shape[1])]) / all_aug_probs.shape[0]) 
                                             if all_aug_probs is not None else None,
                'per_sample_metrics': {
                    'variance': np.concatenate(per_sample_metrics['variance']) if per_sample_metrics['variance'] else None,
                    'disagreement': np.concatenate(per_sample_metrics['disagreement']) 
                                    if per_sample_metrics['disagreement'] else None
                }
            })
    
        return results


class TTAExperiment:
    def __init__(self, config):
        self.config = config
        self.seed_manager = SeedManager()
        self.domain_names = DOMAIN_NAMES['PACS']

        self.all_results = defaultdict(lambda: defaultdict(dict))

        os.makedirs(config['output_dir'], exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(
            config['output_dir'],
            f"tta_results_all_domains_{timestamp}.txt"
        )

        """
        self.stats_root_path = config['stats_root_path']
        self.test_domain = config['test_domain']
        self.modus = config['modus']

        _, self.test_loader = get_dataset(
            name=self.config['dataset_name'],
            root_dir=self.config['data_dir'],
            test_domain=self.config['test_domain']
        )

        self.class_names = self._get_class_names()
        self.domain_names = self._get_domain_names()
        """


    def _get_class_names(self) -> List[str]:
        """Extrahiere Klassennamen aus dem Dataset"""
        try:
            return self.test_loader.dataset.classes
        except AttributeError:
            return [f"Class_{i}" for i in range(self.config['num_classes'])]
        
    @staticmethod
    def generate_mode_variants(domain_names, base_mode):
        """Generiert alle Varianten für einen gegebenen Modus"""
        variants = []
        num_domains = len(domain_names)
    
        if base_mode == 'single':
            variants = [f'single_{i}' for i in range(num_domains)]
        elif base_mode == 'selective':
            # Alle möglichen Domain-Kombinationen (ohne Wiederholung und Reihenfolge)
            variants = [f'selective_{i}_{j}' for i in range(num_domains) 
                       for j in range(i+1, num_domains)]
        elif base_mode == 'average':
            variants = ['average']
    
        return variants

    
    def run_all_domains(self):
        #Applies TTA for all domains as test domains #whack
        with open(self.results_file, 'w') as f:
            f.write("TTA Experiment Results - All Domains\n")
            f.write("="*50 + "\n\n")

            domain_results = {domain: {} for domain in self.domain_names}

            for test_domain in self.domain_names:
                f.write(f"\n\n=== TEST DOMAIN: {test_domain.upper()} ===\n")
                self.config['test_domain'] = test_domain

                target_domains = [d for d in self.domain_names if d != test_domain]
                self.config['target_domains'] = target_domains
                
                for base_mode in self.config['modes']:
                    mode_variants = TTAExperiment.generate_mode_variants(self.domain_names, base_mode)           

                    for mode in mode_variants:
                        f.write(f"\nMode: {mode.upper()}\n")
                        self.config['mode'] = mode

                        for seed in self.config['seeds']:
                            f.write(f"\nSeed: {seed}\n")
                            f.write("-"*30 + "\n")

                            try:
                                seed_results = self.run_single_seed(seed, test_domain)
                            
                                for target_domain, metrics in seed_results.items():
                                    f.write(f"Target {target_domain}:\n")
                                    f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                                    f.write(f"  Variance: {metrics['variance'].mean():.4f}\n")
                                    f.write(f"  Disagreement: {metrics['disagreement'].mean():.4f}\n")
                                self.all_results[test_domain][mode][seed] = seed_results
                            except Exception as e:
                                #f.write(f"Error for mode {mode}, seed {seed}: {str(e)}\n")
                                import traceback
                                error_msg = f"Error for mode {mode}, seed {seed}:\n{str(e)}\n{traceback.format_exc()}"
                                f.write(error_msg + "\n")
                                print(error_msg)
                                if self.config['verbose']:
                                    print(f"Error occurred for mode {mode}, seed {seed}:")
                                    print(str(e))

        json_file = self.results_file.replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        return self.all_results
    
    """
    def run_all_domains(self):
        #Applies TTA for all domains as test domains with combined functionality
        # Prepare both text and JSON outputs
        with open(self.results_file, 'w') as txt_f, \
             open(self.results_file.replace('.txt', '.json'), 'w') as json_f:
        
            # Write text header
            txt_f.write("TTA Experiment Results - All Domains\n")
            txt_f.write("="*50 + "\n\n")
        
            all_modi = ['average', 'selective'] + [f'single_{i}' for i in range(4)]

            for test_domain in self.domain_names:
                txt_f.write(f"\n\n=== TEST DOMAIN: {test_domain.upper()} ===\n")
                self.config['test_domain'] = test_domain
                self.config['target_domains'] = [d for d in self.domain_names if d != test_domain]

                for mode in all_modi:
                    txt_f.write(f"\nMode: {mode.upper()}\n")
                    self.config['mode'] = mode

                    for seed in self.config['seeds']:
                        txt_f.write(f"\nSeed: {seed}\n")
                        txt_f.write("-"*30 + "\n")

                        try:
                            # This now uses the improved predict() method
                            results = self.run_single_config(test_domain, mode, seed)
                            self.all_results[test_domain][mode][seed] = results
                        
                            # Write to text file in your preferred format
                            txt_f.write(f"Results:\n")
                            txt_f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
                        
                            if 'variance' in results:
                                txt_f.write(f"  Avg Variance: {np.mean(results['variance']):.4f}\n")
                                txt_f.write(f"  Avg Disagreement: {np.mean(results['disagreement']):.4f}\n")

                            txt_f.write(f"  Augmentation Domains: {results.get('augmentation_domains', [])}\n")

                        except Exception as e:
                            error_msg = f"Error for mode {mode}, seed {seed}: {str(e)}"
                            txt_f.write(error_msg + "\n")
                            if self.config['verbose']:
                                print(error_msg)
                            continue

            # Save complete results to JSON
            json.dump(self.all_results, json_f, indent=2)

        return self.all_results

    def run_single_config(self, test_domain: str, mode: str, seed: int) -> dict:
        Improved version that works with your predict() method
        self.seed_manager.set_seed(seed)
    
        # Load model (unchanged from your original)
        model_path = os.path.join(
            self.config['models_root_path'],
            f"seed_{seed}",
            f"best_fold_{test_domain}.pt"
        )
    
        model = resnet50(
            pretrained=False,
            num_classes=self.config['num_classes'],
            num_domains=len(self.domain_names),
            domain_names=self.domain_names,
            use_mixstyle=False
        ).to(self.config['device'])
    
        checkpoint = torch.load(model_path, map_location=self.config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
    
        # Get test loader
        _, _, test_loader = get_dataset(
            name=self.config['dataset_name'],
            root_dir=self.config['data_dir'],
            test_domain=test_domain
        )
    
        # Initialize TTA classifier with your predict() method
        tta = TTAClassifier(
            model=model,
            stats_root_path=self.config['models_root_path'],
            test_domain=test_domain,
            device=self.config['device'],
            num_classes=self.config['num_classes'],
            mode=mode,
            seed=seed,
            class_names=test_loader.dataset.classes,
            domain_names=self.domain_names,
            vis_dir=self.config['output_dir']
        )
    
        # Returns results in your structure
        return tta.predict(test_loader, return_uncertainty=True)
    """


    def run_single_seed(self, seed, test_domain):
        """Run full pipeline for one seed"""
        self.seed_manager.set_seed(seed)

        #model = resnet50(pretrained=True, num_classes=self.config['num_classes']).to(self.config['device'])
        model_path = os.path.join(
            self.config['models_root_path'],
            f"seed_{seed}",
            f"best_fold_{test_domain}.pt"
        )

        checkpoint = torch.load(model_path, map_location=self.config['device'])

        #resnet hier pretrained=False oder pretrained=True??
        model = resnet50(
            pretrained=False, 
            num_classes=self.config['num_classes'], 
            #num_domains=len(DOMAIN_NAMES[self.config['dataset']]),
            num_domains=len(self.domain_names),
            domain_names=self.domain_names,
            use_mixstyle=False
            ).to(self.config['device'])
        #model.load_state_dict(torch.load(model_path))
        #model = model.to(self.config['device'])
        #model.load_state_dict(checkpoint['model_state_dict'])
        model_state_dict = checkpoint['model_state_dict']
        style_stats_state = checkpoint['style_stats']

        converted_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('mixstyle.'):
                new_key = k.replace('mixstyle.style_stats.', 'style_stats.')
                converted_state_dict[new_key] = v
            else:
                converted_state_dict[k] = v

        model.load_state_dict(converted_state_dict, strict=False)

        if 'style_stats' in checkpoint:
            style_stats = checkpoint['style_stats']
            converted_style_stats = {}
            for k, v in style_stats.items():
                new_key = k.replace('mixstyle.style_stats.', 'style_stats')
                converted_style_stats[new_key] = v
            model.style_stats.load_state_dict(converted_style_stats, strict=False)

        print(f"\nInitializing run for seed {seed}, test domain {test_domain}")
        print(f"test domain {test_domain}")

        
        try:
            test_domain_idx = self.domain_names.index(test_domain)
        except ValueError:
            available = ", ".join(self.domain_names)
            raise ValueError(f"Unknown test domain '{test_domain}'. Available: {available}")
        

        # 3. Dataset und Loader erstellen
        try:
            dataset = get_dataset(
                name=self.config['dataset_name'],
                root_dir=self.config['data_dir'],
                test_domain=test_domain_idx  # Hier den Index übergeben
            )

            #_, _, test_dataset = dataset.generate_lodo_splits()

            splits = dataset.generate_lodo_splits()
            _, _, test_dataset = splits[test_domain_idx]

            test_loader = DataLoader(
                test_dataset,
                batch_size=8,
                shuffle=False,
                collate_fn=lambda b: (
                    torch.stack([x[0] for x in b]),  # images
                    torch.tensor([x[1] for x in b]),  # labels
                    torch.tensor([x[2] for x in b]) if len(b[0]) > 2 else None  # domains
                )
            )

            # Batch mit Index 5 anzeigen
            #batch_idx = 5
            #specific_batch = list(test_loader)[batch_idx]
            #print(f"Batch {batch_idx}: {specific_batch}")

            if test_loader is None:
                raise ValueError("test_loader is None! Überprüfen Sie self.test_domain in generate_loaders().")
            print("Test loader exists:", test_loader is not None)
            print("Successfully created test loader")
        except Exception as e:
            print(f"Error creating dataset/loader: {str(e)}")
            raise

        """
        # 4. Batch testen
        try:
            print("vor first batch")
            print("Test loader exists vor first batch:", test_loader is not None)
            print("Anzahl der Batches im Test-Loader:", len(test_loader))
            first_batch = next(iter(test_loader))
            for batch in first_batch:
                print("Got batch")
                print("Type:", type(batch))
                print("Len:", len(batch))
                for i, b in enumerate(batch):
                    print(f"Batch[{i}]:", type(b), getattr(b, 'shape', b))
                break
            #images, labels, domains = first_batch  # Annahme: 3-elementiger Tuple
            
            images, labels = first_batch
            # Shapes anzeigen
            print("Images shape:", images.shape)  # Sollte [Batch, Channels, H, W] sein (z.B. [32, 3, 224, 224])
            print("Labels shape:", labels.shape)  # Sollte [Batch] (z.B. [32])
            #print("Domains shape:", domains.shape)

            print("nach first batch")
            print(f"Batch contains {len(first_batch)} elements")
            if len(first_batch) >= 2:
                print(f"First element shape: {first_batch[0].shape}")
                print(f"Second element shape: {first_batch[1].shape}")
            if len(first_batch) >= 3:
                print(f"First element shape: {first_batch[0].shape}")
                print(f"Second element shape: {first_batch[1].shape}")
                print(f"Third element type: {type(first_batch[2])}")
            
        except Exception as e:
            print(f"Error checking first batch: {str(e)}")
            raise
        """
        

        tta = TTAClassifier(
            model=model,
            stats_root_path=self.config['models_root_path'],
            test_domain=test_domain,
            mode=self.config['mode'],
            device=self.config['device'],
            num_classes=self.config['num_classes'],
            verbose=self.config['verbose'],
            seed=seed,
            class_names=self._get_class_names(),  # Übergebe Klassennamen
            domain_names=self.domain_names,
            vis_dir=self.config['output_dir']
        )

        results = {}
        target_domains = [d for d in self.domain_names if d != test_domain]

        for target_domain in target_domains:
            if self.config['verbose']:
                print(f"Test Domain: {test_domain} | Seed {seed} | Mode {self.config['mode']} | Target Domain {target_domain}")
            
            results[target_domain] = tta.predict(
                test_loader,
                num_augments=self.config['num_augments']
            )

        self.all_results[test_domain][self.config['mode']][seed] = results
        
        return results
    

    def run_multiple_seeds(self):
        "Main method to run across all seeds"
        return self.run_all_domains()
    

    def aggregate_results(self, results):
        """Calculate statistics across seeds"""
        aggregated = {}
        domains = list(results[next(iter(results))].keys())

        for domain in domains:
            accuracies = [results[seed][domain]['accuracy'] for seed in results]
            aggregated[domain] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'all_accuracies': accuracies,
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            }
        
        return aggregated
    
    def save_results(self, results):
        """Save results to file"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tta_results_{timestamp}.json"

        with open(os.path.join(self.config['output_dir'], filename), 'w') as f:
            json.dump(results, f, indent=2)


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TTA Experiment Pipeline')
    
    # Basic paths
    parser.add_argument('--models_root_path', type=str, default='./experiments/train_results/test/saved_models', 
                        help='Root path to trained models (contains seed_X folders)')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/hahlers/datasets', 
                        help='Root directory for dataset')
    parser.add_argument('--dataset', type=str, default='PACS', 
                        help='Dataset name: PACS, VLCS')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes for used Dataset; PACS: 7, VLCS: 5')
    
    # Experiment parameters
    parser.add_argument('--num_augments', type=int, default=3, 
                       help='Number of augmentations per sample')
    parser.add_argument('--modes', nargs='+', type=str, default=['single', 'selective', 'average'],
                        help='Modes to run: single, selective, average')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 0],
                       help='Random seeds to run')
    parser.add_argument('--output_dir', type=str, default='./experiments/test_results', 
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    return {
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'dataset': args.dataset,
        'dataset_name': args.dataset,
        'data_dir': args.data_dir,
        'models_root_path': args.models_root_path,
        'stats_root_path': args.models_root_path,
        'data_dir': args.data_dir,
        'num_augments': args.num_augments,
        'seeds': args.seeds,
        'output_dir': args.output_dir,
        'verbose': args.verbose,
        'num_classes': args.num_classes,
        'modes': args.modes
    }


def main():
    config = parse_args()
    
    experiment = TTAExperiment(config)
    final_results = experiment.run_multiple_seeds()
    
    print("\n=== Final Results Across Seeds ===")
    for test_domain, mode_results in final_results.items():
        for mode, seed_results in mode_results.items():
            for seed, domain_results in seed_results.items():
                print(f"Test Domain: {test_domain} | Mode: {mode} | Seed {seed}")
                for domain, metrics in domain_results.items():
                    print(f"   Target domain: {domain}: Accuracy={metrics['accuracy']:.4%}")


if __name__ == "__main__":
    main()