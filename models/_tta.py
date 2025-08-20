import os
import torch
import numpy as np
import random
import json
import argparse
import torch.nn as nn
import datetime
import traceback
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Any
from torch.utils.data import DataLoader
import logging
from torchmetrics import Accuracy
from collections import defaultdict
from ._styleextraction import StyleStatistics, DomainAwareHook
from ._resnet import resnet50
from data._datasets import get_dataset, DOMAIN_NAMES, CLASS_NAMES
from utils._visualize import Visualizer


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to serializable formats"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    else:
        return obj


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
        self.base_mode = None
        self._parse_mode()

        self.test_domain = test_domain
        self.stats_root_path = stats_root_path
        
        self.seed = seed if seed is not None else 42
        self.seed_manager = SeedManager(self.seed)
        self.seed_manager.set_seed()
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing TTAClassifier with verbose logging")

        # freeze model and setup
        self.model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        
        # ensure model is in eval mode (important for BatchNorm/Dropout)
        self.model.eval()
        
        self.feature_extractor = model.to(device)
        
        if num_classes is not None:
            feature_dim = self._get_feature_dim()
            self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        else:
            self.classifier = None
            
        self.softmax = nn.Softmax(dim=1)
        
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(self.device)

        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(num_classes)]
        self.domain_names = domain_names #if domain_names is not None else [f"domain_{i}" for i in range(4)]

        self.style_stats = self._load_style_stats()

        self.visualizer = Visualizer(
            class_names=self.class_names,
            domain_names=self.domain_names,
            config={},
            vis_dir=vis_dir
        )
        
        # Hook for feature extraction
        self.features = {}
        self._register_hooks()
        
        # for uncertainty estimation
        self.entropy = nn.CrossEntropyLoss(reduction='none')
        
        # for weighted average aggregation
        self.domain_weights = None
        if self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            self._init_domain_weights()


    def _parse_mode(self):
        if self.mode.startswith("single"):
            self.base_mode = "single"
            #self.target_layers = [0, 1, 2, 3]
            self.target_layers = [int(self.mode.split("_")[1])]
        elif self.mode.startswith("selective"):
            self.base_mode = "selective"
            self.target_layers = [int(x) for x in self.mode.split("_")[1:]]
        elif self.mode == "average":
            self.base_mode = "average"
            self.target_layers = [0, 1, 2, 3]


    def _load_style_stats(self) -> StyleStatistics:
        """Load style statistics for all training domains"""

        stats = StyleStatistics(
            num_domains=len(self.domain_names),
            num_layers=4,
            domain_names=self.domain_names,
            mode=self.base_mode,
            layer_config=self.target_layers if self.base_mode == "selective" else None,
            device=self.device
        )
        
        for domain in [d for d in self.domain_names if d != self.test_domain]:
            stats_path = os.path.join(
                self.stats_root_path,
                f"seed_{self.seed}",
                "style_stats",
                f"test_{self.test_domain}",
                self.mode,
                f"style_stats_{domain}_{self.mode}.pth"
            )
            
            if os.path.exists(stats_path):
                print(f"\nLoading stats for {domain} from {stats_path}")
                domain_stats = torch.load(stats_path, map_location=self.device, weights_only=True)
            
                """
                print("Keys in loaded file:", list(domain_stats.keys()))
                for k, v in domain_stats.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: shape={v.shape} mean={v.mean().item()}")
                    elif isinstance(v, dict):
                        for sk, sv in v.items():
                            if isinstance(sv, torch.Tensor):
                                print(f"  {sk}: shape={sv.shape} mean={sv.mean().item()}")
                """
            
                stats.load_state_dict(domain_stats, strict=False)
            
            elif self.verbose:
                print(f"Warning: No stats found for domain {domain} at {stats_path}")
        
        return stats


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


    def predict(self, 
           test_dataloader: torch.utils.data.DataLoader, 
           viz_dataloader: torch.utils.data.DataLoader,
           num_augments: int = 3, 
           return_uncertainty: bool = False) -> dict:
        """
        Perform TTA prediction with hooks for real-time feature transformation.
    
        Args:
            dataloader: DataLoader with test data
            num_augments: Number of augmentations per sample (ignored in hook-based approach)
            return_uncertainty: Whether to return uncertainty estimates
    
        Returns:
            dict: Contains accuracy, probabilities, uncertainty metrics etc. for each target domain
        """
        self.model.eval()
        self.model.to(self.device)

        available_domains = [d for d in self.domain_names if d != self.test_domain]

        # initialize storage for each target domain
        domain_results = {
            domain: {
                'all_preds': [],
                'all_labels': [],
                'all_probs': [],
                'real_class_probs': [],
                'augmentation_probs': [],
                'variance': [],
                'real_class_var': [],
                'disagreement': []
            } for domain in available_domains
        }

        # for the original predictions (no augmentation)
        original_results = {
            'all_preds': [],
            'all_labels': [],
            'all_probs': [],
            'real_class_probs': []
        }

        viz_data = {
            'raw_training_data': defaultdict(list),
            'features': {
                'original': None, 
                'augmented': defaultdict(list)
            },
            'sample_imgs': {'original': None, 'augmented': {}},
            'probs': {'original': [], 'augmented': defaultdict(list)},
            'first_batch_processed': False
        }

        cross_domain_variance_all, cross_domain_disagreement_all = [], []
        cross_domain_real_class_var_all = []

        try:
            with torch.no_grad():
                for batch in viz_dataloader:
                    images = batch[0].cpu().numpy()
                    domains = batch[2].cpu().numpy() if len(batch) > 2 else None

                    if domains is not None:
                        for domain_idx in np.unique(domains):
                            domain_name = self.domain_names[domain_idx]
                            mask = domains == domain_idx
                            viz_data['raw_training_data'][domain_name].append(
                                images[mask].reshape(len(images[mask]), -1)
                            )
                    
                for domain in viz_data['raw_training_data']:
                    viz_data['raw_training_data'][domain] = np.concatenate(
                        viz_data['raw_training_data'][domain], axis=0)

                batch_idx = 0
                for batch in test_dataloader:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device) if len(batch) > 1 else None
                    _ = batch[2].to(self.device) if len(batch) > 2 else None

                    # 1. Original prediction (no augmentation)
                    orig_logits = self.model(images)
                    orig_probs = self.softmax(orig_logits)
                    orig_preds = torch.argmax(orig_probs, dim=1)

                    if batch_idx == 0:
                        viz_data['features']['original'] = orig_logits #TODO?
                        viz_data['sample_imgs']['original'] = images[0]
                        viz_data['probs']['original'] = orig_probs.cpu().numpy()
                
                    if labels is not None:                        
                        original_results['all_preds'].append(orig_preds.cpu().numpy())
                        original_results['all_labels'].append(labels.cpu().numpy())
                        original_results['all_probs'].append(orig_probs.cpu().numpy())
                        labels_np = labels.cpu().numpy()
                        orig_real_class_probs = orig_probs.cpu().numpy()[np.arange(len(labels_np)), labels_np]
                        original_results['real_class_probs'].append(orig_real_class_probs)

                    #batch_aug_preds = []
                    #probs_per_augmentation = []
                    probs_per_augmentation, preds_per_augmentation = [], []
                    real_class_probs_per_domain = []

                    # 2. process each target domain separately
                    for target_domain in available_domains:
                        hooks = []
                        
                        try:
                            for layer_idx in self.target_layers:
                                hook = DomainAwareHook(
                                    stats_root=os.path.join(
                                        self.stats_root_path,
                                        f"seed_{self.seed}",
                                        "style_stats",
                                        f"test_{self.test_domain}",
                                        self.mode
                                    ),
                                    test_domain=self.test_domain,
                                    target_domain=target_domain,
                                    modus=self.mode,
                                    device=self.device,
                                    layer_idx=layer_idx
                                )
                                for name, module in self.model.named_modules():
                                    if f"layer{layer_idx+1}" in name:
                                        hook = module.register_forward_hook(hook)
                                        hooks.append(hook)
                                        break
                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing domain {target_domain}: {str(e)}")
                            continue

                        # Forward pass with domain adaptation
                        domain_logits = self.model(images)
                        probs_tensor = self.softmax(domain_logits)
                        domain_preds = torch.argmax(probs_tensor, dim=1)
                        domain_probs = probs_tensor.cpu().numpy()
                                        
                        probs_per_augmentation.append(domain_probs)
                        preds_per_augmentation.append(domain_preds.cpu().numpy())

                        # Store predictions and labels for this domain
                        domain_results[target_domain]['all_preds'].append(domain_preds.cpu().numpy())
                        domain_results[target_domain]['all_probs'].append(domain_probs)
                        
                        if labels is not None:
                            labels_np = labels.cpu().numpy()
                            real_class_probs = domain_probs[np.arange(len(labels_np)), labels_np]
                            domain_results[target_domain]['real_class_probs'].append(real_class_probs)
                            real_class_probs_per_domain.append(real_class_probs)
                            domain_results[target_domain]['all_labels'].append(labels_np)

                        if batch_idx == 0:
                            viz_data['features']['augmented'][target_domain].append(domain_logits) #TODO?
                            viz_data['sample_imgs']['augmented'][target_domain] = images[0]
                            viz_data['probs']['augmented'][target_domain].append(domain_probs)
                                                
                        # Remove hooks
                        for hook in hooks:
                            hook.remove()
                    
                    if len(probs_per_augmentation) > 0 and len(preds_per_augmentation) > 0:
                        all_probs = np.stack(probs_per_augmentation, axis=0)
                        
                        per_class_var = np.var(all_probs, axis=0)
                        per_sample_var = per_class_var.mean(axis=1)
                        cross_domain_variance_all.extend(per_sample_var.tolist())
                        
                        if len(real_class_probs_per_domain) > 0:
                            real_class_probs_arr = np.stack(real_class_probs_per_domain, axis=0)
                            real_class_var = np.var(real_class_probs_arr, axis=0)
                            cross_domain_real_class_var_all.extend(real_class_var.tolist())

                        preds_arr = np.stack(preds_per_augmentation, axis=0)
                        batch_disagreements = []
                        for j in range(preds_arr.shape[1]):
                            sample_preds = preds_arr[:, j]
                            unique, counts = np.unique(sample_preds, return_counts=True)
                            disagreement = 1 - (np.max(counts) / len(sample_preds))
                            batch_disagreements.append(disagreement)
                        cross_domain_disagreement_all.extend(batch_disagreements)

                        if not viz_data['first_batch_processed']:
                            viz_data['first_batch_processed'] = True
                
                        batch_idx += 1

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
        finally:
            for module in self.model.modules():
                if hasattr(module, '_forward_hooks'):
                    module._forward_hooks.clear()

        
        mean_cross_variance = float(np.mean(cross_domain_variance_all)) if cross_domain_variance_all else None
        mean_cross_disagreement = float(np.mean(cross_domain_disagreement_all)) if cross_domain_disagreement_all else None
        mean_cross_real_class_var = float(np.mean(cross_domain_real_class_var_all)) if cross_domain_real_class_var_all else None
        
        results = {
            'original': {
                'accuracy': None,
                'predictions': None,
                'labels': None,
                'real_class_probs': None,
                'real_class_var': None
            },
            'target_domains': {}
        }

        if len(original_results['all_labels']) > 0:
            orig_preds = np.concatenate(original_results['all_preds'])
            orig_labels = np.concatenate(original_results['all_labels'])
            orig_acc = (orig_preds == orig_labels).mean()
            orig_real_class_probs = np.concatenate(original_results['real_class_probs'])
            orig_real_class_var = np.var(orig_real_class_probs)            
            
            results['original'] = {
                'accuracy': float(orig_acc),
                'predictions': orig_preds.tolist(),
                'labels': orig_labels.tolist(),
                'probs': np.concatenate(original_results['all_probs']).tolist(),
                'real_class_probs': orig_real_class_probs.tolist(),
                'real_class_var': float(orig_real_class_var)
            }

        # process each target domain
        for domain in available_domains:
            if domain_results[domain] and domain_results[domain]['all_labels']:
                preds = np.concatenate(domain_results[domain]['all_preds'])
                labels = np.concatenate(domain_results[domain]['all_labels'])
                probs = np.concatenate(domain_results[domain]['all_probs'])
                real_class_probs = np.concatenate(domain_results[domain]['real_class_probs'])
            
                acc = (preds == labels).mean()
                #real_class_var = np.var(real_class_probs)
                all_probs = np.concatenate(domain_results[domain]['all_probs'], axis=0)
                if len(all_probs) > 0:
                    per_class_var = np.var(all_probs, axis=0)
                    avg_variance = float(np.mean(per_class_var))
                    real_class_var = float(np.var(real_class_probs))
                else:
                    avg_variance = None
                    real_class_var = None

                #avg_variance = np.mean(domain_results[domain]['variance']) if domain_results[domain]['variance'] else None
                avg_disagreement = np.mean(domain_results[domain]['disagreement']) if domain_results[domain]['disagreement'] else None

                results['target_domains'][domain] = {
                    'accuracy': float(acc),
                    'predictions': preds.tolist(),
                    'labels': labels.tolist(),
                    'probs': probs.tolist(),
                    'real_class_probs': real_class_probs.tolist(),
                    'real_class_var': real_class_var,
                    'variance': avg_variance if avg_variance is not None else None,
                    'disagreement': float(avg_disagreement) if avg_disagreement is not None else None,
                    'test_domain': self.test_domain,
                    'mode': self.mode,
                    'seed': self.seed,
                    'cross_domain_mean_variance': mean_cross_variance,
                    'cross_domain_mean_disagreement': mean_cross_disagreement,
                    'cross_domain_mean_real_class_var': mean_cross_real_class_var
                }
        
        domain_accuracies = [
            results['target_domains'][domain]['accuracy']
            for domain in available_domains
            if 'accuracy' in results['target_domains'][domain]
        ]
        accuracy_variance = float(np.var(domain_accuracies)) if len(domain_accuracies) >= 2 else None

        results['metadata'] = {
            'accuracy_variance': accuracy_variance,
            'test_domain': self.test_domain,
            'augmentation_domains': available_domains,
            'mode': self.mode,
            'seed': self.seed,
            'device': self.device
        }

        results['cross_domain_mean_variance'] = mean_cross_variance
        results['cross_domain_mean_disagreement'] = mean_cross_disagreement
        results['cross_domain_mean_real_class_var'] = mean_cross_real_class_var

        if self.visualizer and viz_data['first_batch_processed']:
            try:
                """
                self.visualizer.visualize_tta_tsne(
                    original_features=viz_data['features']['original'],
                    augmented_features={k: torch.cat(v) for k, v in viz_data['features']['augmented'].items()},
                    raw_train_data=viz_data['raw_training_data'],
                    test_domain=self.test_domain
                )
                
                self.visualizer.visualize_tta_gradcam(
                    model=self.model,
                    original_img=viz_data['sample_imgs']['original'],
                    augmented_imgs=viz_data['sample_imgs']['augmented']
                )
                """
                self.visualizer.plot_tta_confusion_matrices(
                    original_results={
                        'preds': np.concatenate(original_results['all_preds']),
                        'labels': np.concatenate(original_results['all_labels'])
                    },
                    augmented_results={
                        domain: {
                            'preds': np.concatenate(data['all_preds']),
                            'labels': np.concatenate(data['all_labels'])
                        }
                        for domain, data in domain_results.items()
                    }
                )
                self.visualizer.plot_confidence_intervals({
                    'target_domains': {
                        domain: {
                            'all_probs': np.concatenate(data['all_probs'])
                        }
                        for domain, data in domain_results.items()
                    }
                })
                """
                if len(viz_data['sample_imgs']['original'].shape) == 3:
                    self.visualizer.plot_feature_stats_heatmap(
                        original_img=viz_data['sample_imgs']['original'],
                        augmented_imgs=list(viz_data['sample_imgs']['augmented'].values())
                    )
                
                self.visualizer.plot_prediction_consistency(
                    original_probs=np.concatenate(viz_data['probs']['original']),
                    augmented_probs=[
                        np.concatenate(probs)
                        for probs in viz_data['probs']['augmented'].values()
                    ]
                )
                """
            except Exception as e:
                print(f"Visualization failed: {str(e)}")
                if self.verbose:
                    traceback.print_exc()

        return results


class TTAExperiment:
    def __init__(self, config):
        self.config = config
        self.seed_manager = SeedManager()
        self.domain_names = DOMAIN_NAMES['PACS']
        self.class_names = CLASS_NAMES['PACS']

        self.all_results = defaultdict(lambda: defaultdict(dict))

        os.makedirs(config['output_dir'], exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(
            config['output_dir'],
            f"tta_results_all_domains_{timestamp}.txt"
        )


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
            variants = [f'selective_{i}_{j}' for i in range(num_domains) 
                       for j in range(i+1, num_domains)]
        elif base_mode == 'average':
            variants = ['average']
    
        return variants
    
    
    def get_all_modes(self):
        return [
            "single_0", "single_1", "single_2", "single_3",
            "selective_0_1", "selective_0_2", "selective_0_3",
            "selective_1_2", "selective_1_3", "selective_2_3",
            "average"
        ]
    
    """
    def get_all_modes(self):
        return ["selective_0_1", "single_0", "average"]
    """
        
    def run_all_domains(self):
        """Run TTA for all domains as test domains, maintaining both text logs and JSON results"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"tta_results_all_domains_{timestamp}"

        text_file = os.path.join(self.config['output_dir'], f"{base_filename}.txt")
        json_file = os.path.join(self.config['output_dir'], f"{base_filename}.json")

        results = {
            'metadata': {
                'timestamp': timestamp,
                'config': self.config,
                'domain_names': self.domain_names
            },
            'results': {}
        }

        with open(text_file, 'w') as txt_f:
            txt_f.write("TTA Experiment Results\n")
            txt_f.write("="*50 + "\n\n")
            txt_f.write(f"Configuration:\n{json.dumps(self.config, indent=2)}\n\n")

            for test_domain in self.domain_names:
                txt_f.write(f"\n\n=== TEST DOMAIN: {test_domain.upper()} ===\n")
                self.config['test_domain'] = test_domain
                results['results'][test_domain] = {}

                for mode in self.get_all_modes():
                    txt_f.write(f"\nMode: {mode.upper()}\n")
                    self.config['mode'] = mode
                    results['results'][test_domain][mode] = {}

                    target_accs = defaultdict(list)
                    target_real_class_vars = defaultdict(list)

                    for seed in self.config['seeds']:
                        txt_f.write(f"\nSeed: {seed}\n")
                        txt_f.write("-"*30 + "\n")

                        try:
                            seed_results = self.run_single_seed(seed, test_domain)
                            results['results'][test_domain][mode][str(seed)] = seed_results

                            if 'original' in seed_results:
                                orig = seed_results['original']
                                txt_f.write("Target original:\n")
                                txt_f.write(f"  Accuracy: {orig['accuracy']:.4f}\n")
                                if orig.get('real_class_variance') is not None:
                                    txt_f.write(f"  Real class variance: {orig['real_class_variance']:.4f}\n")

                            for target_domain in self.domain_names:
                                if target_domain == test_domain:
                                    pass
                                if target_domain in seed_results and target_domain != test_domain:
                                    metrics = seed_results[target_domain]
                                    txt_f.write(f"Target {target_domain}:\n")
                                    if 'accuracy' in metrics:
                                        target_accs[target_domain].append(metrics['accuracy'])
                                        txt_f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                                    if metrics.get('real_class_variance') is not None:
                                        target_real_class_vars[target_domain].append(metrics['real_class_variance'])
                                        txt_f.write(f"  Real class variance: {metrics['real_class_variance']:.4f}\n")
                            
                            cross = seed_results.get('cross_domain')
                            if cross:
                                txt_f.write("Cross-domain summary (variation across domains):\n")
                                if cross.get('mean_variance_across_domains') is not None:
                                    txt_f.write(f"  Mean cross-domain variance: {cross['mean_variance_across_domains']:.4f}\n")
                                if cross.get('mean_disagreement_across_domains') is not None:
                                    txt_f.write(f"  Mean cross-domain disagreement: {cross['mean_disagreement_across_domains']:.4f}\n")
                                if cross.get('mean_real_class_variance') is not None:
                                    txt_f.write(f"  Mean real class variance: {cross['mean_real_class_variance']:.4f}\n")

                        except Exception as e:
                            error_msg = f"Error for mode {mode}, seed {seed}: {str(e)}"
                            txt_f.write(error_msg + "\n")
                            results['results'][test_domain][mode][str(seed)] = {
                                'error': error_msg,
                                'traceback': traceback.format_exc()
                            }
                            if self.config['verbose']:
                                print(error_msg)

                    if target_accs:
                        all_accs = []
                        all_real_class_vars = []

                        for acc_list in target_accs.values():
                            all_accs.extend(acc_list)

                        if all_accs:
                            mean_acc = np.mean(all_accs)
                            std_acc = np.std(all_accs)
                            txt_f.write(f"\nMean accuracy across seeds (target domains only): {mean_acc:.4f} ± {std_acc:.4f}\n")
                            results['results'][test_domain][mode]['mean_target_accuracy'] = float(mean_acc)

                        if all_real_class_vars:
                            mean_var = np.mean(all_real_class_vars)
                            std_var = np.std(all_real_class_vars)
                            txt_f.write(f"Mean real class variance across seeds (target domains only): {mean_var:.4f} ± {std_var:.4f}\n")
                            results['results'][test_domain][mode]['mean_real_class_variance'] = float(mean_var)

        with open(json_file, 'w') as f:
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=2)

        return results


    def run_single_seed(self, seed, test_domain):
        """Run full pipeline for one seed"""
        self.seed_manager.set_seed(seed)

        model_path = os.path.join(
            self.config['models_root_path'],
            f"seed_{seed}",
            f"best_fold_{test_domain}.pt"
        )

        checkpoint = torch.load(model_path, map_location=self.config['device'], weights_only=True)

        #resnet hier pretrained=False oder pretrained=True??
        model = resnet50(
            pretrained=False, 
            num_classes=self.config['num_classes'], 
            num_domains=len(self.domain_names),
            domain_names=self.domain_names,
            use_mixstyle=False
            ).to(self.config['device'])
        
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
            
        try:
            test_domain_idx = self.domain_names.index(test_domain)
        
            viz_dataset, _, test_dataset = get_dataset(
                name=self.config['dataset_name'],
                root_dir=self.config['data_dir'],
                test_domain=test_domain_idx
            ).generate_lodo_splits()[test_domain_idx]

            test_loader = DataLoader(
                test_dataset,
                batch_size=8,
                shuffle=False,
                collate_fn=lambda b: (
                    torch.stack([x[0] for x in b]).to(self.config['device']),  # images
                    torch.tensor([x[1] for x in b]).to(self.config['device']),  # labels
                    torch.tensor([x[2] for x in b]).to(self.config['device']) if len(b[0]) > 2 else None  # domains
                )
            )

            viz_loader = DataLoader(
                viz_dataset,
                batch_size=32,
                shuffle=False, #TODO true oder false?
                collate_fn=lambda b: (
                    torch.stack([x[0] for x in b]).to(self.config['device']),  # images
                    torch.tensor([x[1] for x in b]).to(self.config['device']),  # labels
                    torch.tensor([x[2] for x in b]).to(self.config['device']) if len(b[0]) > 2 else None  # domains
                )
            )

        except Exception as e:
            print(f"Error creating dataset/loader: {str(e)}")
            raise
        
        tta = TTAClassifier(
            model=model,
            stats_root_path=self.config['models_root_path'],
            test_domain=test_domain,
            mode=self.config['mode'],
            device=self.config['device'],
            num_classes=self.config['num_classes'],
            verbose=self.config['verbose'],
            seed=seed,
            class_names=self.class_names,
            domain_names=self.domain_names,
            #vis_dir=self.config['output_dir']
            vis_dir=os.path.join(self.config['output_dir'], f"seed_{seed}", test_domain)
        )

        results = tta.predict(
            test_loader,
            viz_loader,
            num_augments=self.config['num_augments']
        )

        processed_results = {
            'original': {
                'accuracy': results['original']['accuracy'],
                'variance': results['original'].get('variance', None),
                'disagreement': results['original'].get('disagreement', None),
                'real_class_variance': results['original'].get('real_class_variance', None)
            },
            'cross_domain': {
                'mean_variance_across_domains': results.get('cross_domain_mean_variance'),
                'mean_disagreement_across_domains': results.get('cross_domain_mean_disagreement'),
                'mean_real_class_variance': results.get('cross_domain_mean_real_class_var')
            }
        }
    
        for domain, metrics in results['target_domains'].items():
            processed_results[domain] = {
                'accuracy': metrics['accuracy'],
                'variance': metrics['variance'],
                'disagreement': metrics['disagreement'],
                'real_class_variance': metrics.get('real_class_var'),
                'real_class_probs': metrics.get('real_class_probs')
            }
        
        return processed_results
    

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
    parser.add_argument('--models_root_path', type=str, default='./experiments/train_results/pacs_woMS/saved_models', 
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
    experiment.run_multiple_seeds()

    print("=== TTA Complete ===")
    

if __name__ == "__main__":
    main()