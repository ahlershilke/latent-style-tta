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
        self.domain_names = domain_names

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
        

    def _parse_mode(self):
        if self.mode.startswith("single"):
            self.base_mode = "single"
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
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass"""
        features = self.feature_extractor(x)
        if self.classifier:
            return self.classifier(features)
        return features


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
        per_sample_uncertainty, per_sample_correct = [], []
        sample_count = 0

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

                    # original prediction (no augmentation)
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
                        correct_vec = (orig_preds == labels).long().cpu().numpy()

                    probs_per_augmentation, preds_per_augmentation = [], []
                    real_class_probs_per_domain = []

                    # process each target domain separately
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

                        # forward pass with domain adaptation
                        domain_logits = self.model(images)
                        probs_tensor = self.softmax(domain_logits)
                        domain_preds = torch.argmax(probs_tensor, dim=1)
                        domain_probs = probs_tensor.cpu().numpy()
                                        
                        probs_per_augmentation.append(domain_probs)
                        preds_per_augmentation.append(domain_preds.cpu().numpy())

                        # store predictions and labels for this domain
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
                                                
                        # remove hooks
                        for hook in hooks:
                            hook.remove()
                    
                    if len(probs_per_augmentation) > 0 and len(preds_per_augmentation) > 0:
                        all_probs = np.stack(probs_per_augmentation, axis=0)
                        
                        per_class_var = np.var(all_probs, axis=0)
                        per_sample_var = per_class_var.mean(axis=1)
                        cross_domain_variance_all.extend(per_sample_var.tolist())

                        per_sample_correct.extend(correct_vec.tolist())
                        
                        if len(real_class_probs_per_domain) > 0:
                            real_class_probs_arr = np.stack(real_class_probs_per_domain, axis=0)
                            real_class_var = np.var(real_class_probs_arr, axis=0)
                            cross_domain_real_class_var_all.extend(real_class_var.tolist())

                            per_sample_uncertainty.extend(real_class_var.tolist())

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
                all_probs = np.concatenate(domain_results[domain]['all_probs'], axis=0)
                if len(all_probs) > 0:
                    per_class_var = np.var(all_probs, axis=0)
                    avg_variance = float(np.mean(per_class_var))
                    real_class_var = float(np.var(real_class_probs))
                else:
                    avg_variance = None
                    real_class_var = None

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

        # random drop curve
        if per_sample_uncertainty and per_sample_correct:
            scores = np.asarray(per_sample_uncertainty)
            correct = np.asarray(per_sample_correct).astype(np.float32)
            N = len(scores)

            drop_grid = np.arange(0,95+1,5)

            sort_idx = np.argsort(scores)[::-1]
            correct_sorted = correct[sort_idx]
            curve_uncert = []

            for p in drop_grid:
                k = int(np.floor(p / 100.0 * N))
                remain = correct_sorted[k:]
                acc = float(remain.mean()) if remain.size > 0 else np.nan
                curve_uncert.append(acc)

            n_trials = getattr(self, 'random_trials', None) or 1000
            rng = np.random.default_rng(seed=self.seed)

            rand_accs_per_percent = {int(p): [] for p in drop_grid.tolist()}

            for _ in range(n_trials):
                for p in drop_grid:
                    perm = rng.permutation(N)
                    k = int(np.floor((p / 100.0) * N))
                    if k == 0:
                        keep_ids = perm
                    else:
                        keep_ids = perm[k:]

                    acc = float(correct[keep_ids].mean()) if keep_ids.size > 0 else np.nan
                    rand_accs_per_percent[int(p)].append(acc)
            
            curve_rand_mean = [float(np.nanmean(rand_accs_per_percent[int(p)])) for p in drop_grid]
            curve_rand_std = [float(np.nanstd(rand_accs_per_percent[int(p)])) for p in drop_grid]
            
            results['drop_curve'] = {
                'percent': drop_grid.tolist(),
                'uncertainty_curve': curve_uncert,
                'random_within1sigma_mean': curve_rand_mean,
                'random_within1sigma_std': curve_rand_std,
                'random_samples': {int(p): [float(a) for a in rand_accs_per_percent[int(p)]] for p in drop_grid},
                'random_trials': int(n_trials)
            }
            results['per_sample_uncertainty'] = per_sample_uncertainty
            results['per_sample_correct'] = per_sample_correct

        return results


class TTAExperiment:
    def __init__(self, config):
        self.config = config
        self.seed_manager = SeedManager()
        self.domain_names = DOMAIN_NAMES[config['dataset']]
        self.class_names = CLASS_NAMES[config['dataset']]

        self.all_results = defaultdict(lambda: defaultdict(dict))

        os.makedirs(config['output_dir'], exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(
            config['output_dir'],
            f"tta_results_all_domains_{timestamp}.txt"
        )

        self.visualizer = Visualizer(
            class_names=self.class_names,
            domain_names=self.domain_names,
            config={},
            vis_dir=self.config['output_dir']
        )


    def _get_class_names(self) -> List[str]:
        """Extract Class names from the dataset"""
        try:
            return self.test_loader.dataset.classes
        except AttributeError:
            return [f"Class_{i}" for i in range(self.config['num_classes'])]
          
    
    def get_all_modes(self):
        return [
            "single_0", "single_1", "single_2", "single_3",
            "selective_0_1", "selective_0_2", "selective_0_3",
            "selective_1_2", "selective_1_3", "selective_2_3",
            "average"
        ]
    
        
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

                    all_drop_curves = []
                    for seed in self.config['seeds']:
                        seed_res = results['results'][test_domain][mode].get(str(seed), {})
                        if 'drop_curve' in seed_res:
                            all_drop_curves.append(seed_res['drop_curve'])

                    if all_drop_curves:
                        drop_grid = all_drop_curves[0]['percent']
                        
                        unc_curves = np.array([dc['uncertainty_curve'] for dc in all_drop_curves])
                        rand_mean_curves = np.array([dc['random_within1sigma_mean'] for dc in all_drop_curves])
                        rand_std_curves = np.array([dc['random_within1sigma_std'] for dc in all_drop_curves])

                        mean_unc = unc_curves.mean(axis=0)
                        mean_rand = np.nanmean(rand_mean_curves, axis=0)
                        var_means = np.nanvar(rand_mean_curves, axis=0, ddof=0)
                        mean_seed_vars = np.nanmean(rand_std_curves**2, axis=0)
                        pooled_std = np.sqrt(var_means + mean_seed_vars)

                        self.visualizer.plot_uncertainty_dropping(
                            drop_grid=drop_grid,
                            curve_uncert=mean_unc.tolist(),
                            curve_rand_mean=mean_rand.tolist(),
                            curve_rand_std=pooled_std.tolist(),
                            test_domain=test_domain,
                            mode=mode
                        )

                        p = np.asarray(drop_grid, dtype=np.float64) / 100.0

                        auc_mean_curve = float(np.trapezoid(mean_unc, p))
                        auc_random_mean_curve = float(np.trapezoid(mean_rand, p))
                        gain_abs_mean_curve = float(auc_mean_curve - auc_random_mean_curve)
                        gain_rel_mean_curve = (
                            float(gain_abs_mean_curve / auc_random_mean_curve)
                            if auc_random_mean_curve != 0 else np.nan
                        )
                        auc_unc_seeds = [float(np.trapezoid(curve, p)) for curve in unc_curves]
                        auc_rand_seeds = [float(np.trapezoid(curve, p)) for curve in rand_mean_curves]
                        gain_abs_seeds = [u - r for u, r in zip(auc_unc_seeds, auc_rand_seeds)]
                        gain_rel_seeds = [
                            (g / r) if r != 0 else np.nan
                            for g, r in zip(gain_abs_seeds, auc_rand_seeds)
                        ]

                        mean_auc = float(np.mean(auc_unc_seeds))
                        std_auc = float(np.std(auc_unc_seeds))
                        mean_gain_abs = float(np.mean(gain_abs_seeds))
                        std_gain_abs = float(np.std(gain_abs_seeds))

                        results['results'][test_domain][mode]['drop_curve_metrics'] = {
                            'auad_from_mean_curve': auc_mean_curve,
                            'auad_random_from_mean_curve': auc_random_mean_curve,
                            'auad_gain_abs_from_mean_curve': gain_abs_mean_curve,
                            'auad_gain_rel_from_mean_curve': gain_rel_mean_curve,
                            'auad_mean': mean_auc,
                            'auad_std': std_auc,
                            'auad_gain_abs_mean': mean_gain_abs,
                            'auad_gain_abs_std': std_gain_abs,
                        }

                        # write to txt file
                        txt_f.write("\nDrop-curve metrics (averaged across seeds):\n")
                        txt_f.write(f"  AUAD (uncertainty curve): {mean_auc:.4f} ± {std_auc:.4f}\n")
                        txt_f.write(f"  Gain over random (abs):   {mean_gain_abs:.4f} ± {std_gain_abs:.4f}\n")
                        txt_f.write(f"  AUAD from mean curve:     {auc_mean_curve:.4f}\n")
                        txt_f.write(f"  Gain from mean curve:     {gain_abs_mean_curve:.4f}\n")

                    
                    all_u, all_c = [], []
                    auroc_per_seed = []

                    for seed in self.config['seeds']:
                        seed_res = results['results'][test_domain][mode].get(str(seed), {})
                        u = seed_res.get('per_sample_uncertainty')
                        c = seed_res.get('per_sample_correct')
                        if u is None or c is None:
                            continue
                        u = np.asarray(u, dtype=np.float64)
                        c = np.asarray(c, dtype=np.int32)

                        is_error = 1 - c
                        auc_seed = self._auroc_from_scores_numpy(u, is_error)
                        if np.isfinite(auc_seed):
                            auroc_per_seed.append(auc_seed)
                        all_u.append(u)
                        all_c.append(c)

                    if len(all_u) > 0:
                        u_pooled = np.concatenate(all_u, axis=0)
                        c_pooled = np.concatenate(all_c, axis=0)
                        auroc_pooled = self._auroc_from_scores_numpy(u_pooled, 1-c_pooled)
                    else:
                        auroc_pooled = float('nan')
                    
                    if auroc_per_seed:
                        auroc_mean = float(np.mean(auroc_per_seed))
                        auroc_std = float(np.std(auroc_per_seed))
                    else:
                        auroc_mean = float('nan')
                        auroc_std = float('nan')

                    results['results'][test_domain][mode]['variance_auroc'] = {
                        'per_seed': auroc_per_seed,
                        'mean': auroc_mean,
                        'std': auroc_std,
                        'pooled': auroc_pooled
                    }
                    
                    txt_f.write("\nVariance as error detector (AUROC):\n")
                    txt_f.write(f"  AUROC (per-seed): mean {auroc_mean:.4f} ± {auroc_std:.4f}\n")
                    txt_f.write(f"  AUROC (pooled over seeds): {auroc_pooled:.4f}\n")                 

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
                shuffle=False,
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

        if 'drop_curve' in results:
            processed_results['drop_curve'] = results['drop_curve']

        if 'per_sample_uncertainty' in results and 'per_sample_correct' in results:
            processed_results['per_sample_uncertainty'] = results['per_sample_uncertainty']
            processed_results['per_sample_correct'] = results['per_sample_correct']
    
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

    
    @staticmethod
    def _auroc_from_scores_numpy(scores: np.ndarray, is_error: np.ndarray) -> float:
        """
        AUROC between scores and binary labels (is_error: 1=incorrect, 0=correct).
        Tie-aware via average ranks. Returns np.nan if only one class present.
        """
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(is_error, dtype=np.int32)
        P = int((y == 1).sum())
        N = int((y == 0).sum())
        if P == 0 or N == 0:
            return float('nan')

        # ranks with tie handling (average ranks)
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

        # average ranks for ties
        s_sorted = scores[order]
        i = 0
        while i < len(scores):
            j = i + 1
            while j < len(scores) and s_sorted[j] == s_sorted[i]:
                j += 1
            if j - i > 1:
                avg = (i + 1 + j) / 2.0  # average of (i+1) .. j
                ranks[order[i:j]] = avg
            i = j

        # Mann–Whitney U formulation
        rank_pos_sum = ranks[y == 1].sum()
        auc = (rank_pos_sum - P * (P + 1) / 2.0) / (P * N)
        return float(auc)


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TTA Experiment Pipeline')
    
    # Basic paths
    parser.add_argument('--models_root_path', type=str, default='./experiments/train_results/vlcs_woMS/saved_models', 
                        help='Root path to trained models (contains seed_X folders)')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/hahlers/datasets', 
                        help='Root directory for dataset')
    parser.add_argument('--dataset', type=str, default='PACS', 
                        help='Dataset name: PACS, VLCS')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes for used Datasets; PACS: 7, VLCS: 5')
    
    # Experiment parameters
    parser.add_argument('--num_augments', type=int, default=3, 
                       help='Number of augmentations per sample (number of training domains)')
    parser.add_argument('--modes', nargs='+', type=str, default=['single', 'selective', 'average'],
                        help='Modes to run: single, selective, average')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 0],
                       help='Random seeds to run')
    parser.add_argument('--output_dir', type=str, default='./experiments/test_results', 
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--random_trials', type=int, default=1000,
                    help='Number of random extractions for the random baseline for drop curves')
    parser.add_argument('--random_band', type=str, default='none', choices=['sigma','none'],
                    help='Restrict random removals to ±1σ band (sigma) or over all samples (none)')

    
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
        'modes': args.modes,
        'random_trials': args.random_trials,
        'random_band': args.random_band
    }


def main():
    config = parse_args()
    
    experiment = TTAExperiment(config)
    experiment.run_multiple_seeds()

    print("=== TTA Complete ===")
    

if __name__ == "__main__":
    main()