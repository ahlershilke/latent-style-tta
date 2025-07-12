import os
import torch
import numpy as np
import random
import json
import argparse
from sklearn.metrics import accuracy_score
import torch.nn as nn
import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from torchmetrics import Accuracy
from collections import defaultdict
from ._styleextraction import StyleStatistics
from ._resnet import resnet50
from data._datasets import get_dataset
from _styleextraction import StyleStatistics


class AggregationStrategy(Enum):
    AVERAGE = auto()
    MAJORITY_VOTE = auto()
    MAX_CONFIDENCE = auto()
    WEIGHTED_AVERAGE = auto()

class FeatureAggregationStrategy(Enum):
    MEAN = auto()
    MAX = auto()
    CONCAT = auto()
    #ATTENTION = auto()

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
                 stats_path: Optional[str] = None, 
                 device: str = 'cuda', 
                 num_classes: Optional[int] = None,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.AVERAGE,
                 feature_aggregation: FeatureAggregationStrategy = FeatureAggregationStrategy.MEAN,
                 verbose: bool = False,
                 seed: Optional[int] = None
                 ):
        """
        Improved TTA Classifier with multiple enhancements.
        
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

        self.seed_manager = SeedManager(seed if seed is not None else 42)
        self.seed_manager.set_seed()
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing TTAClassifier with verbose logging")

        # Freeze model and setup
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
        
        # Load statistics if provided
        self.style_stats = StyleStatistics(
            num_domains=len(model.style_stats.domain_stats),  # Anzahl Domänen
            num_layers=len(model.style_stats.layer_stats)#,    # Anzahl Layer
            #mode="average"                                    # oder "selective"
        )
        
        if stats_path:
            self.style_stats.load_state_dict(torch.load(stats_path))
        
        # Hook for feature extraction
        self.features = {}
        self._register_hooks()
        
        # For uncertainty estimation
        self.entropy = nn.CrossEntropyLoss(reduction='none')
        
        # For weighted average aggregation
        self.domain_weights = None
        if self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            self._init_domain_weights()


    def _load_stats(self, stats_path: str):
        """Load statistics with validation"""
        stats = torch.load(stats_path)
        self.domain_stats = stats.get('domain', {})
        self.layer_stats = stats.get('layer', {})
        
        if self.verbose:
            self.logger.info(f"Loaded stats with {len(self.domain_stats)} domains and {len(self.layer_stats)} layers")
            for layer, domains in self.layer_stats.items():
                self.logger.info(f"Layer {layer} has stats for domains: {list(domains.keys())}")


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
                          target_domain: int) -> torch.Tensor:
        """
        Apply cross-domain feature augmentation using pre-saved style statistics.
    
        Args:
            features: Input features to augment (shape: [B, C, H, W])
            layer_idx: Layer index to identify which statistics to use
            target_domain: Target domain index for cross-domain mixing
        
        Returns:
            Augmented features with cross-domain mixed statistics
        """
        B = features.size(0)
        device = features.device
    
        # 1. Compute current features statistics
        mu = features.mean(dim=[2, 3], keepdim=True)
        var = features.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
    
        # 2. Normalize features
        features_normed = (features - mu) / sig
    
        # 3. Load pre-saved statistics for target domain
        stats_path = os.path.join(
            "~/experiments/train_results/saved_models/style_stats", #TODO path womöglich nicht ganz richtig
            f"test_{target_domain}.pth" #TODO pfad anpassen!
        )
    
        domain_stats = torch.load(stats_path, map_location=device)
    
        # Get stats for current layer
        #layer_stats = domain_stats[layer_idx] #TODO try this
        #target_mu = layer_stats['mu'].to(device)
        #target_sig = layer_stats['sig'].to(device)
        target_mu = torch.from_numpy(domain_stats[layer_idx]['mu']).to(device)
        target_sig = torch.from_numpy(domain_stats[layer_idx]['sig']).to(device)
    
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


    def predict(self, 
               dataloader: torch.utils.data.DataLoader, 
               num_augments: int = 4,
               return_uncertainty: bool = False) -> dict:
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
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        uncertainties = []

        all_augmentation_probs = []  # To store probs from each augmentation
        per_sample_metrics = {
            'variance': [],
            'disagreement': []
        }
        
        available_domains = list({d for layer in self.layer_stats.values() for d in layer.keys()})
        
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
                        augmented = self._augment_features(orig_features, layer_name, domain=domain)
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
    

class TTAExperiment:
    def __init__(self, config):
        self.config = config
        self.seed_manager = SeedManager()

    def run_single_seed(self, seed):
        """Run full pipeline for one seed"""
        self.seed_manager.set_seed(seed)

        model = resnet50(pretrained=True, num_classes=self.config['num_classes']).to(self.config['device'])

        tta = TTAClassifier(
            model=model,
            stats_path=self.config['stats_path'],
            device=self.config['device'],
            num_classes=self.config['num_classes'],
            verbose=self.config['verbose'],
            seed=seed
        )

        _, test_loader = get_dataset(
            name=self.config['dataset_name'],
            root_dir=self.config['data_dir'],
            test_domain=self.config['test_domain']
        )

        results = {}
        for target_domain in self.config['target_domains']:
            if self.config['verbose']:
                print(f"Seed {seed} | Target domain {target_domain}")
            
            results[target_domain] = tta.predict(
                test_loader,
                num_augments=self.config['num_augments']
            )
        
        return results
    
    def run_multiple_seeds(self):
        "Main method to run across all seeds"
        all_results = {}

        for seed in self.config['seeds']:
            all_results[seed] = self.run_single_seed(seed)

        final_results = {}
        self.save_results(final_results)

        return final_results
    
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

"""
def main():
    # 1. Modell und Daten laden
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(pretrained=False, num_classes=7).to(device)
    
    # 2. Statistik-Datei laden (simuliert)
    stats_path = "style_stats.pth"
    style_stats = StyleStatistics(num_domains=4, num_layers=4)
    torch.save(style_stats.state_dict(), stats_path)  # Demo: Erstelle leere Stats

    # 3. TTA-Classifier initialisieren
    tta = TTAClassifier(
        model=model,
        stats_path=stats_path,
        device=device,
        num_classes=7,
        verbose=True
    )

    # 4. Testdaten laden (Beispiel mit PACS)
    _, test_loader = get_dataset(
        name="PACS",
        root_dir="./data",
        test_domain=1  # z.B. 1="cartoon" als Testdomäne
    )

    # 5. Vorhersage mit Style-Transfer zu Domain 2 ("sketch")
    accuracy, probs = tta.predict(
        test_loader,
        target_domain=2,  # Ziel-Style: "sketch"
        num_augments=1
    )
    
    print(f"Accuracy mit Sketch-Style-Augmentierung: {accuracy:.2%}")

if __name__ == "__main__":
    main()
"""

def parse_args() -> Dict[str, Any]:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TTA Experiment Pipeline')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='PACS', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory for data')
    parser.add_argument('--test_domain', type=int, default=1, help='Test domain index')
    parser.add_argument('--target_domains', nargs='+', type=int, default=[0, 2, 3], 
                       help='List of target domains for style transfer')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--stats_path', type=str, default='style_stats.pth', 
                       help='Path to style statistics file')
    
    # Experiment parameters
    parser.add_argument('--num_augments', type=int, default=5, 
                       help='Number of augmentations per sample')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 101112],
                       help='Random seeds to run')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    
    # Output control
    parser.add_argument('--output_dir', type=str, default='./results', 
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    return {
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'dataset_name': args.dataset,
        'data_dir': args.data_dir,
        'test_domain': args.test_domain,
        'target_domains': args.target_domains,
        'num_classes': args.num_classes,
        'stats_path': args.stats_path,
        'num_augments': args.num_augments,
        'seeds': args.seeds,
        'output_dir': args.output_dir,
        'verbose': args.verbose,
        'num_workers': args.num_workers
    }

def main():
    config = parse_args()
    
    experiment = TTAExperiment(config)
    final_results = experiment.run_multiple_seeds()
    
    # Print summary
    print("\n=== Final Results Across Seeds ===")
    for domain, metrics in final_results.items():
        print(f"Domain {domain}: {metrics['mean_accuracy']:.2%} ± {metrics['std_accuracy']:.2%}")
        print(f"   Range: {metrics['min']:.2%} - {metrics['max']:.2%}")
        print(f"   All values: {[f'{x:.2%}' for x in metrics['all_accuracies']]}")

if __name__ == "__main__":
    main()