import torch
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn

class TTAClassifier(nn.Module):
    def __init__(self, model, stats_path, device='cuda', num_classes=None):
        """
        Args:
            model: Pre-trained model with feature extractor and classifier
            stats_path: Path to .pth file containing μ and σ stats
            device: Device to run computations on
        """
        super.__init__(self)
        self.device = device

        self.model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        
        self.feature_extractor = model.to(device)
        
        # Add linear classifier if needed
        if num_classes is not None:
            feature_dim = self._get_feature_dim()
            self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        else:
            self.classifier = None
            
        self.softmax = nn.Softmax(dim=1)
        
        #TODO ich glaube nicht, dass die extraktion so einfach ist, da die layer-stats in den domain-stats liegen
        # zumindest sieht das in der json so aus
        # Load statistics if provided
        if stats_path:
            stats = torch.load(stats_path)
            self.domain_stats = stats.get('domain', {})
            self.layer_stats = stats.get('layer', {})
            self._register_hooks()
        else:
            self.domain_stats = {}
            self.layer_stats = {}
        
        # Hook for feature extraction
        self.features = {}
        self._register_hooks()
        

    def _register_hooks(self):
        """Register hooks to capture features at specific layers"""
        self.features = {}
        for layer_name in self.layer_stats.keys():
            layer = dict([*self.feature_extractor.named_modules()])[layer_name]
            
            def hook_fn(module, input, output, name=layer_name):
                self.features[name] = output
            
            #TODO register_forward_hook
            #layer.register_forward_hook(hook_fn)
            layer.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, layer_name))
    

    #TODO
    def _augment_features(self, features, layer_name, domain='target'):
        """
        Apply feature-level augmentation using stored μ and σ
        """
        if layer_name not in self.layer_stats:
            return features
            
        # Get stats for this layer and domain
        stats = self.layer_stats[layer_name].get(domain, {})
        if not stats:
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
        return augmented
    

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.classifier:
            return self.classifier(features)
        return features
    

    def predict(self, dataloader, num_augments=4):
        """
        Perform feature-level TTA prediction
        
        Args:
            dataloader: DataLoader with test data
            num_augments: Number of augmentations per sample
            
        Returns:
            tuple: (accuracy, all_probs) where all_probs contains class probabilities
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                labels = batch[1] if len(batch) > 1 else None
                
                batch_probs = []
                
                # Original pass
                logits = self(images)
                probs = self.softmax(logits)
                batch_probs.append(probs)
                
                # Augmented passes
                for _ in range(num_augments):
                    # Forward pass to capture features
                    _ = self.feature_extractor(images)
                    
                    # Augment features
                    augmented_features = {}
                    for layer_name in self.features.keys():
                        orig = self.features[layer_name]
                        augmented = self._augment_features(orig, layer_name)
                        augmented_features[layer_name] = augmented
                    
                    # Forward with augmented features
                    if hasattr(self.feature_extractor, 'forward_with_features'):
                        #TODO forward_with_features
                        features = self.feature_extractor.forward_with_features(images, augmented_features)
                    else:
                        features = self.feature_extractor(images)
                        
                    if self.classifier:
                        logits = self.classifier(features)
                    else:
                        logits = features
                        
                    batch_probs.append(self.softmax(logits))
                
                # Average predictions
                avg_probs = torch.mean(torch.stack(batch_probs), dim=0)
                preds = torch.argmax(avg_probs, dim=1)
                
                all_probs.append(avg_probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
                if labels is not None:
                    all_labels.append(labels.numpy())
        
        # Convert results
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs, axis=0)
        
        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels)
            accuracy = accuracy_score(all_labels, all_preds)
            return accuracy, all_probs
        return None, all_probs