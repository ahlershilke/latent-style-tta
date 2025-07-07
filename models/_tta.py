import torch
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple
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

class TTAClassifier(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 stats_path: Optional[str] = None, 
                 device: str = 'cuda', 
                 num_classes: Optional[int] = None,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.AVERAGE,
                 feature_aggregation: FeatureAggregationStrategy = FeatureAggregationStrategy.MEAN,
                 verbose: bool = False
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
            num_layers=len(model.style_stats.layer_stats),    # Anzahl Layer
            mode="average"                                    # oder "selective"
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
    #(self, features: torch.Tensor, layer_name: str, domain: str = 'target') -> torch.Tensor:
        """
        Apply feature-level augmentation using stored μ and σ with validation.
        
        Args:
            features: Input features to augment
            layer_name: Which layer these features come from
            domain: Which domain to augment towards
            
        Returns:
            Augmented features
        ###
        if layer_name not in self.layer_stats:
            if self.verbose:
                self.logger.warning(f"No stats found for layer {layer_name}")
            return features
            
        # Get stats for this layer and domain
        stats = self.layer_stats[layer_name].get(domain, {})
        if not stats:
            if self.verbose:
                self.logger.warning(f"No stats found for domain {domain} in layer {layer_name}")
            return features
            
        mu = stats.get('mu', torch.zeros_like(features)).to(features.device)
        sigma = stats.get('sigma', torch.ones_like(features)).to(features.device)
        
        # Normalize features
        eps = 1e-5
        mu_current = features.mean(dim=[2, 3], keepdim=True)
        sigma_current = (features.var(dim=[2, 3], keepdim=True) + eps).sqrt()
        normalized = (features - mu_current) / sigma_current
        
        # Denormalize with target stats
        augmented = normalized * sigma + mu
        
        if self.verbose:
            self.logger.info(f"Augmented features for layer {layer_name} towards domain {domain}")
            
        return augmented
        """

        # Hole μ/σ aus StyleStatistics
        mu_target, sig_target = self.style_stats.get_style_stats(target_domain)
        mu_target = mu_target[layer_idx].to(features.device)  # [C, 1, 1]
        sig_target = sig_target[layer_idx].to(features.device)
    
        # Normalisierung + Augmentierung
        mu_current = features.mean(dim=[2, 3], keepdim=True)
        sig_current = features.std(dim=[2, 3], keepdim=True)
    
        return (features - mu_current) / (sig_current + 1e-5) * sig_target + mu_target


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
               return_uncertainty: bool = False) -> Union[Tuple[float, np.ndarray], 
                                                         Tuple[float, np.ndarray, np.ndarray]]:
        """
        Perform feature-level TTA prediction with enhanced functionality.
        
        Args:
            dataloader: DataLoader with test data
            num_augments: Number of augmentations per sample
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            tuple: (accuracy, all_probs) or (accuracy, all_probs, uncertainties)
                   where all_probs contains class probabilities and uncertainties
                   contains entropy values for each prediction
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        uncertainties = []
        
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

                # Original (non-augmented)
                logits = self(images)
                probs = self.softmax(logits)
                batch_probs.append(probs)
                batch_logits.append(logits)

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

                # Aggregate predictions according to strategy
                avg_probs = self._aggregate_predictions(batch_probs)
                preds = torch.argmax(avg_probs, dim=1)

                all_probs.append(avg_probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

                if return_uncertainty:
                    # Calculate entropy as uncertainty measure
                    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1)
                    uncertainties.append(entropy.cpu().numpy())

                if labels is not None:
                    all_labels.append(labels.numpy())
                    # Update torchmetrics accuracy
                    self.accuracy.update(preds, torch.tensor(labels, device=self.device))

        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs, axis=0)
        
        if return_uncertainty:
            uncertainties = np.concatenate(uncertainties)

        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels)
            accuracy = self.accuracy.compute().item()
            
            if return_uncertainty:
                return accuracy, all_probs, uncertainties
            return accuracy, all_probs
        
        if return_uncertainty:
            return None, all_probs, uncertainties
        return None, all_probs
    

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