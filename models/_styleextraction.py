import os
import torch
import torch.nn as nn
import json
import warnings
from typing import List, Dict


class StyleStatistics(nn.Module):
    DEFAULT_MOMENTUM = 0.9
    WARMUP_UPDATES = 100

    def __init__(self,
        num_domains: int,
        num_layers: int,
        domain_names: List[str] = None,
        mode: str = None,       # "average", "selective", "single"
        layer_config=None,      # for "selective" or "single"
        use_ema: bool = True,  # exponential moving average
        ema_momentum: float = DEFAULT_MOMENTUM,  # momentum for EMA
        device="cuda",
        **kwargs
    ):
        
        """Initialize StyleStatistics module for tracking and mixing feature statistics.
        Tracks mean (mu) and standard deviation (sig) statistics per domain and layer,
        with different mixing strategies controlled by the mode parameter.

        params:
            num_domains (int): Number of distinct style domains to track.
            num_layers (int): Total number of network layers where MixStyle could be applied.
            mode (str): Style mixing mode, one of:
                - "single": Use statistics from single target layer.
                - "average": Average statistics across all layers.
                - "selective": Use only specified layers (requires layer_config).
                - "paired": Mix between layer pairs (requires layer_config).
            layer_config: Configuration specific to mode:
                - For "selective": List of layer indices to use (e.g., [0, 2]).
                - For "paired": List of layer index pairs (e.g., [(0,1), (1,2)]).
                - For "single": Can specify target layer via kwargs.
            use_ema (bool): Whether to use exponential moving average for updates.
            ema_momentum (float): Momentum factor for EMA updates (0-1).
            **kwargs: Additional mode-specific arguments:
                - For "single": target_layer (int) can specify which layer to use.

        Raises:
            ValueError: If invalid mode is specified or required layer_config is missing
        """

        super().__init__()
        self.num_domains = num_domains
        self.num_layers = num_layers
        self.mode = mode
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.device = torch.device(device)

        self.domain_names = domain_names if domain_names is not None else [f"domain_{i}" for i in range(num_domains)]

        self.register_buffer('layer_counts', 
                             torch.zeros(num_domains, num_layers, dtype=torch.long))
        self.register_buffer('count', torch.zeros(num_domains, dtype=torch.long))

        # initialize storage structures
        self.mu_dict = nn.ParameterDict()  # {layer_idx: parameter[num_domains, C, 1, 1]}
        self.sig_dict = nn.ParameterDict()
        #self.register_buffer('count', torch.zeros(num_domains, dtype=torch.long))

        self.target_layer = None
        if self.mode == "selective":
            if not layer_config:
                raise ValueError("layer_config required for selective mode")
            self.target_layer = layer_config
            self.layer_momentum = nn.ParameterDict({
                str(layer): nn.Parameter(torch.tensor(ema_momentum))
                for layer in self.target_layer
            })
        elif self.mode == "single":
            self.target_layer = kwargs.get("target_layer", 0)
        elif self.mode == "average":
            self.target_layer = [0,1,2,3]
        else:
            if mode not in ["selective", "average", "single"]:
                raise ValueError(f"Unknown mode: {mode}")

        # initialize buffers for first use tracking
        self._initialized = True
    

    """
    def _get_dynamic_momentum(self, layer_idx: int, update_count:int) -> float:
        if self.mode == "selective" and str(layer_idx) in self.layer_momentum:
            base_momentum = torch.sigmoid(self.layer_momentum[str(layer_idx)])

        else:
            base_momentum = self.ema_momentum
        
        if update_count < self.WARMUP_UPDATES:
            return base_momentum * (update_count / self.WARMUP_UPDATES)
        
        return base_momentum

    
    def _initialize_buffers(self):
        #Initializes the data structures for storing style statistics.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mu_dict = nn.ParameterDict()
        self.sig_dict = nn.ParameterDict()
    
        self.register_buffer("count", torch.zeros(self.num_domains, dtype=torch.long, device=device))
        self._initialized = True    # to prevent re-initialization
    """


    def _batch_update(self, domain_idx: torch.Tensor, layer_idx: int, 
                      mu: torch.Tensor, sig: torch.Tensor):
        """
        Vektorisiertes Update für Batch-Eingaben
        Args:
            domain_idx: LongTensor [B,] mit Domain-Indizes
            mu: FloatTensor [B,C] oder [B,C,1,1]
            sig: FloatTensor [B,C] oder [B,C,1,1]
        """
        # Input Validation
        assert domain_idx.dim() == 1, "domain_idx must be 1D"
        #print(f"mu.shape[0] {mu.shape[0]}")
        #print(f"domain_idx.shape[0] {domain_idx.shape[0]}")
        assert mu.shape[0] == domain_idx.shape[0], "Batch dimension mismatch"
    
        # Squeeze auf [B,C] falls nötig
        mu = mu.squeeze(-1).squeeze(-1) if mu.dim() == 4 else mu
        sig = sig.squeeze(-1).squeeze(-1) if sig.dim() == 4 else sig
    
        """
        for d in torch.unique(domain_idx):
            mask = (domain_idx == d)
            if mask.sum() == 0:
                continue
            
            # Vektorisiertes Mittelwertberechnung
            #mu_mean = mu[mask].mean(dim=0)  # [C,]
            #sig_mean = sig[mask].mean(dim=0) # [C,]
        
            # EMA Update mit dynamischem Momentum
            self._ema_update(d.item(), layer_idx, mu[mask], sig[mask])
        """
        for domain in domain_idx.unique():
            mask = (domain_idx == domain)
            self._ema_update(domain, layer_idx, mu[mask], sig[mask])

    """
    def _update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        #Update statistics for given domain and layer with mode-specific rules
        if not self._should_update_layer(layer_idx):
            return

        #mu = mu.squeeze(-1).squeeze(-1)  # [B, C]
        #sig = sig.squeeze(-1).squeeze(-1)  # [B, C]

        #B = mu.shape[0]
        #domain_idx ist ein int hier
        if not isinstance(domain_idx, torch.Tensor):
            #domain_idx = torch.tensor(domain_idx, device=mu.device)
            domain_idx = torch.tensor([domain_idx], device=mu.device)

        self._batch_update(domain_idx, layer_idx, mu, sig)
    """
        

    def _should_update_layer(self, layer_idx: int) -> bool:
        """Check if layer should be updated based on current mode."""
        if self.mode == "selective":
            return layer_idx in self.target_layer
        elif self.mode == "single":
            return layer_idx == self.target_layer
        
        return True  # for 'average' and 'attention' modes


    def _get_momentum(self, layer_idx: int, update_count: int) -> float:
        """Get momentum value for specific layer and mode."""
        base_momentum = self.ema_momentum
        if self.mode == "selective" and str(layer_idx) in self.layer_momentum:
            base_momentum = torch.sigmoid(self.layer_momentum[str(layer_idx)])

        if update_count < self.WARMUP_UPDATES:
            return base_momentum * (update_count / self.WARMUP_UPDATES)

        return base_momentum
    

    def _init_layer(self, layer_idx: int, num_channels: int):
        """Initialize buffers for a specific layer"""
    
        self.mu_dict[str(layer_idx)] = nn.Parameter(
            torch.zeros(self.num_domains, num_channels, device=self.device),
            requires_grad=False
        )
        self.sig_dict[str(layer_idx)] = nn.Parameter(
            torch.zeros(self.num_domains, num_channels, device=self.device),
            requires_grad=False
        )


    def _validate_inputs(self, domain_idx, mu, sig):
        if not isinstance(domain_idx, (int, torch.Tensor)):
            raise TypeError("domain_idx must be int or Tensor")
        
        if not isinstance(domain_idx, (int, torch.Tensor)):
            raise TypeError("domain_idx must be int or Tensor")
    
        if mu.shape != sig.shape:
            raise ValueError("mu and sig must have same shape")

        if mu.dim() not in [2,4]:  # [B,C] oder [B,C,1,1]
            raise ValueError("Input must be 2D or 4D tensors")
    

    def _ema_update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        self._validate_inputs(domain_idx, mu, sig)
    
        mu = mu.squeeze(-1).squeeze(-1) if mu.dim() == 4 else mu
        sig = sig.squeeze(-1).squeeze(-1) if sig.dim() == 4 else sig
    
        update_count = self.layer_counts[domain_idx, layer_idx]
        momentum = self._get_momentum(layer_idx, update_count)

        # Channel-Dimension extrahieren
        num_channels = mu.size(1)  # [B, C] → C

        # Initialisiere Layer-Buffer, falls nicht vorhanden oder falsche Dimension
        if str(layer_idx) not in self.mu_dict:
            self._init_layer(layer_idx, num_channels)
        elif self.mu_dict[str(layer_idx)].shape[1] != num_channels:
            # Falls Channel-Dimension nicht übereinstimmt, neu initialisieren
            self._init_layer(layer_idx, num_channels)

        mu_mean = mu.mean(dim=0)  # [C]
        sig_mean = sig.mean(dim=0)  # [C]

        if self.count[domain_idx] == 0:
            self.mu_dict[str(layer_idx)].data[domain_idx] = mu_mean
            self.sig_dict[str(layer_idx)].data[domain_idx] = sig_mean
        else:
            self.mu_dict[str(layer_idx)].data[domain_idx] = (
                momentum * self.mu_dict[str(layer_idx)][domain_idx] + 
                (1 - momentum) * mu_mean
            )
            self.sig_dict[str(layer_idx)].data[domain_idx] = (
                momentum * self.sig_dict[str(layer_idx)][domain_idx] +
                (1 - momentum) * sig_mean
            )
    
        self.count[domain_idx] += 1
        self.layer_counts[domain_idx, layer_idx] += 1


    def _simple_update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        """Perform simple average update."""
        self._validate_inputs(domain_idx, mu, sig)
        total = self.count[domain_idx]
        self.mu_dict[layer_idx, domain_idx] = (self.mu_dict[layer_idx, domain_idx] * total + mu) / (total + 1)
        self.sig_dict[layer_idx, domain_idx] = (self.sig_dict[layer_idx, domain_idx] * total + sig) / (total + 1)
        self.count[domain_idx] += 1

    
    @staticmethod
    def interpolate_to_size(tensor, target_size=256):
        """Interpolates a 1D tensor to target_size using linear interpolation."""
        if len(tensor) == target_size:
            return tensor
        
        return torch.nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='linear',
            align_corners=True
        ).squeeze()


    def get_style_stats(self, domain_idx: int):
        """Returns the style statistics for the given domain index.
    
        Args:
            domain_idx: Index of the target domain.
        
        Returns:
            Tuple of (mu, sig) where:
                - mu: Combined mean statistics
                - sig: Combined standard deviation statistics
        """
        print(f"Current mode: {self.mode}, target_layer: {self.target_layer}")
        print(f"Available layers: {list(self.mu_dict.keys())}")

        
        #if str(self.target_layer) not in self.mu_dict:
            #raise ValueError(f"No style stats collected for target layer {self.target_layer}")

        if self.mode == "single":
            if self.target_layer is None:
                raise ValueError("Target layer is None in single mode")
            
            layer_key = str(self.target_layer)
            if layer_key not in self.mu_dict:
                raise ValueError(f"No style stats collected for layer {self.target_layer}")
            
            print(f"Layer {layer_key} mu stats - min: {self.mu_dict[layer_key][domain_idx].min().item()}, "
              f"max: {self.mu_dict[layer_key][domain_idx].max().item()}")
            print(f"Layer {layer_key} sig stats - min: {self.sig_dict[layer_key][domain_idx].min().item()}, "
              f"max: {self.sig_dict[layer_key][domain_idx].max().item()}")
            
            mu = self.mu_dict[str(self.target_layer)][domain_idx]
            sig = self.sig_dict[str(self.target_layer)][domain_idx]
        
        elif self.mode == "average":
            # averaging channel values per Block
            if not self.mu_dict:
                raise ValueError("No style stats collected for any layer")
            
            mus, sigs = [], []
        
            for layer in sorted(self.mu_dict.keys(), key=int):
                # Get and normalize layer stats
                layer_mu = self.mu_dict[layer][domain_idx]
                layer_sig = self.sig_dict[layer][domain_idx]

                print(f"Layer {layer} mu stats - min: {layer_mu.min().item()}, max: {layer_mu.max().item()}")
                print(f"Layer {layer} sig stats - min: {layer_sig.min().item()}, max: {layer_sig.max().item()}")
            
                layer_mu = (layer_mu - layer_mu.min()) / (layer_mu.max() - layer_mu.min() + 1e-8)
                layer_sig = (layer_sig - layer_sig.min()) / (layer_sig.max() - layer_sig.min() + 1e-8)
            
                # Interpolate to target dimension
                mus.append(self.interpolate_to_size(layer_mu, 256))
                sigs.append(self.interpolate_to_size(layer_sig, 256))
        
            mu = torch.mean(torch.stack(mus), dim=0)
            sig = torch.mean(torch.stack(sigs), dim=0)
        
        elif self.mode == "selective":
            # averaging across selected layers
            mus, sigs = [], []
            
            for layer in self.target_layer:
                layer_key = str(layer)
                if layer_key in self.mu_dict:
                    mus.append(self.mu_dict[layer_key][domain_idx])
                    sigs.append(self.sig_dict[layer_key][domain_idx])
        
            mu = torch.mean(torch.stack(mus), dim=0) if mus else torch.zeros(1)
            sig = torch.mean(torch.stack(sigs), dim=0) if sigs else torch.zeros(1)

            #mu = [self.mu_dict[str(layer)][domain_idx] for layer in self.target_layer]
            #sig = [self.sig_dict[str(layer)][domain_idx] for layer in self.target_layer]

        return mu.unsqueeze(-1).unsqueeze(-1), sig.unsqueeze(-1).unsqueeze(-1)


    def save_style_stats_to_json(self, filepath: str) -> None:
        """
        Saves style stats in a domain-centric flat structure.
        Example output for domain 0, layer 0 with 3 channels:
        {
            "mu": [1.2, 0.5, 0.8],  # Flattened values
            "sig": [0.3, 0.2, 0.4]
        }
        """
        def flatten_stats(tensor):
            """Flattens a tensor to 1D list, removing empty dimensions."""
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu()
            return [float(x) for x in tensor.squeeze().cpu().numpy().ravel()]
        
        """
        stats_dict = {
            "domain_stats": {
                self.domain_names[domain_id]: {  # Use actual domain name here
                    "mu": flatten_stats(self.mu_dict[str(self.target_layer)][domain_id]),
                    "sig": flatten_stats(self.sig_dict[str(self.target_layer)][domain_id]),
                    "count": int(self.count[domain_id].item())
                }
                for domain_id in range(self.num_domains)
            },
            "metadata": {
                "mode": self.mode,
                "target_layer": self.target_layer if hasattr(self, 'target_layer') else None,
                "num_domains": self.num_domains,
                "domain_names": self.domain_names
            }
        }
        

        if self.mode == "single":
            layer_key = str(self.target_layer)
        elif self.mode == "average":
            # For average mode, we need to handle all layers
            layer_key = "0"  # Default to first layer for structure, but handle all in metadata
        else:
            layer_key = "0"  # Fallback
        """
    
        # Prepare base stats structure
        stats_dict = {
            "domain_stats": {},
            "metadata": {
                "mode": self.mode,
                "num_domains": self.num_domains,
                "domain_names": self.domain_names,
                "target_layer": int(self.target_layer) if hasattr(self, 'target_layer') and self.mode != "average" else None
            }
        }

        """
        # Handle different modes
        for domain_id in range(self.num_domains):
            if self.count[domain_id].item() == 0:
                continue  # Skip domains with no data
            
            if self.mode == "single":
                layer_key = str(self.target_layer)
                if layer_key in self.mu_dict:
                    stats_dict["domain_stats"][self.domain_names[domain_id]] = {
                        "mu": flatten_stats(self.mu_dict[layer_key][domain_id]),
                        "sig": flatten_stats(self.sig_dict[layer_key][domain_id]),
                        "count": int(self.count[domain_id].item())
                    }
            
            elif self.mode == "average":
                # For average mode, we need to handle normalization
                mus = []
                sigs = []
            
                for layer in self.mu_dict.keys():
                    layer_mu = self.mu_dict[layer][domain_id]
                    layer_sig = self.sig_dict[layer][domain_id]
                
                    # Normalize each layer's stats
                    layer_mu = (layer_mu - layer_mu.min()) / (layer_mu.max() - layer_mu.min() + 1e-8)
                    layer_sig = (layer_sig - layer_sig.min()) / (layer_sig.max() - layer_sig.min() + 1e-8)
                
                    mus.append(layer_mu)
                    sigs.append(layer_sig)
            
                if mus:  # Only add if we have data
                    avg_mu = torch.mean(torch.stack(mus), dim=0)
                    avg_sig = torch.mean(torch.stack(sigs), dim=0)
                
                    stats_dict["domain_stats"][self.domain_names[domain_id]] = {
                        "mu": flatten_stats(avg_mu),
                        "sig": flatten_stats(avg_sig),
                        "count": int(self.count[domain_id].item())
                    }
        """

        for domain_id in range(self.num_domains):
            if self.count[domain_id].item() == 0:
                print(f"Skipping domain {domain_id} - no data")
                continue
            
            mu, sig = self.get_style_stats(domain_id)
        
            stats_dict["domain_stats"][self.domain_names[domain_id]] = {
                "mu": flatten_stats(mu),
                "sig": flatten_stats(sig),
                "count": int(self.count[domain_id].item())
            }
        
        with open(filepath, "w") as f:
            json.dump(stats_dict, f, indent=2)


    @classmethod
    def load_style_stats_from_json(cls, filepath: str):
        """
        Loads the StyleStatistics from a JSON file.
        Args:
            filepath: Path to the JSON file.
        """
        with open(filepath, "r") as f:
            stats_dict = json.load(f)

        style_stats = cls(
        num_domains=stats_dict["metadata"]["num_domains"],
        num_layers=1,  # Only loading single layer stats now
        mode=stats_dict["metadata"]["mode"],
        target_layer=stats_dict["metadata"]["target_layer"]
    )

        # Get domain names and indices
        domain_names = list(stats_dict["domain_stats"].keys())
    
        # Initialize layer
        target_layer = stats_dict["metadata"]["target_layer"]
        if target_layer is not None:
            # Estimate channel dim from first domain's stats
            sample_mu = stats_dict["domain_stats"][domain_names[0]]["mu"]
            num_channels = len(sample_mu)
            style_stats._init_layer(target_layer, num_channels)

            # Load stats for each domain
            for domain_idx, domain_name in enumerate(domain_names):
                domain_data = stats_dict["domain_stats"][domain_name]
                style_stats.mu_dict[str(target_layer)][domain_idx] = torch.tensor(domain_data["mu"])
                style_stats.sig_dict[str(target_layer)][domain_idx] = torch.tensor(domain_data["sig"])
                style_stats.count[domain_idx] = 1  # Default count
            
        return style_stats
    

    def save_style_stats(self, path) -> None:
        torch.save(self.state_dict(), path)


    #TODO wird aktuell nicht gebraucht
    @classmethod
    def load_style_stats(cls, self, path, device="cuda") -> None:
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)


    def warn_infrequent_updates(self, threshold=10):
        """Shows warnings for seldom used Domains/Layers"""
        for d in range(self.num_domains):
            for l in range(self.num_layers):
                count = self.layer_counts[d, l].item()
                if 0 < count < threshold:
                    warnings.warn(f"Domain {d}, Layer {l} has only {count} updates")

    
    def reset_stats(self):
        """Setzt alle Statistiken zurück"""
        for layer in self.mu_dict:
            self.mu_dict[layer].data.zero_()
            self.sig_dict[layer].data.zero_()
        self.count.data.zero_()
        self.layer_counts.data.zero_()
    
    def get_domain_stats(self, domain_idx: int) -> dict:
        """Gibt Statistiken für eine bestimmte Domain zurück"""
        stats = {}
        for layer in sorted(self.mu_dict.keys()):
            stats[f'layer_{layer}_mu'] = self.mu_dict[layer][domain_idx].cpu().numpy()
            stats[f'layer_{layer}_sigma'] = self.sig_dict[layer][domain_idx].cpu().numpy()
        stats['update_count'] = self.count[domain_idx].item()
        return stats
    
    def get_all_stats(self) -> dict:
        """Gibt alle Statistiken für alle Domains und Layer zurück"""
        all_stats = {}
        for domain in range(self.num_domains):
            all_stats[f'domain_{domain}'] = self.get_domain_stats(domain)
        return all_stats


class StyleExtractorManager:
    """Manages creation and execution of multiple style extractors with different modes."""
    
    def __init__(self, domain_names: List[str], device: torch.device):
        self.domain_names = domain_names
        self.device = device
        self.extractors = self._create_style_extractors()
        

    def _get_default_modes(self) -> Dict:
        """Returns the predefined configuration of extraction modes"""
        return {
            'single_0': {'mode': 'single', 'config': {'target_layer': 0}},
            'single_1': {'mode': 'single', 'config': {'target_layer': 1}},
            'single_2': {'mode': 'single', 'config': {'target_layer': 2}},
            'single_3': {'mode': 'single', 'config': {'target_layer': 3}},
            #'selective_0_1': {'mode': 'selective', 'config': [0, 1]},
            #'selective_0_2': {'mode': 'selective', 'config': [0, 2]},
            #'selective_0_3': {'mode': 'selective', 'config': [0, 3]},
            #'selective_1_2': {'mode': 'selective', 'config': [1, 2]},
            #'selective_1_3': {'mode': 'selective', 'config': [1, 3]},
            #'selective_2_3': {'mode': 'selective', 'config': [2, 3]},
            'average': {'mode': 'average', 'config': None}
        }
        

    def _create_style_extractors(self) -> Dict[str, StyleStatistics]:
        """Creates StyleExtractor instances for all desired modes"""
        modes = self._get_default_modes()
        extractors = {}
        
        """
        for layer_idx in range(4):  # For all 4 ResNet layers
            name = f'single_{layer_idx}'
            extractors[name] = StyleStatistics(
                num_domains=len(self.domain_names),
                num_layers=4,
                domain_names=self.domain_names,
                mode='single',
                target_layer=layer_idx,  # Explicit layer assignment
                use_ema=True,
                ema_momentum=0.9,
                device=self.device
            )
    
        # Create average extractor
        extractors['average'] = StyleStatistics(
            num_domains=len(self.domain_names),
            num_layers=4,
            domain_names=self.domain_names,
            mode='average',
            use_ema=True,
            ema_momentum=0.9,
            device=self.device
        )
        

        for name, config in modes.items():        
            extractors[name] = StyleStatistics(
                num_domains=len(self.domain_names),
                num_layers=4,
                domain_names=self.domain_names,
                mode=config['mode'],
                layer_config=config['config'],
                use_ema=True,
                ema_momentum=0.9,
                device=self.device
                #target_layer=target_layer
            )
        """

        for name, config in modes.items():
            # Extrahiere die Parameter für den aktuellen Modus
            mode = config['mode']
            layer_config = config['config']
        
            # Spezielle Behandlung für single-Modus
            if mode == 'single':
                kwargs = {'target_layer': int(layer_config['target_layer'])} # layer_config['target_layer']
            else:
                kwargs = {'layer_config': layer_config}
        
            extractors[name] = StyleStatistics(
                num_domains=len(self.domain_names),
                num_layers=4,
                domain_names=self.domain_names,
                mode=mode,
                use_ema=True,
                ema_momentum=0.9,
                device=self.device,
                **kwargs
            )
        
        return extractors
    

    def save_style_stats(self, domain_name: str, results_dir: str = "style_stats") -> None:
        """Saves all collected style statistics"""
        stats_dir = os.path.join(results_dir, f"test_{domain_name}")
        os.makedirs(stats_dir, exist_ok=True)
    
        for mode_name, extractor in self.extractors.items():
            mode_dir = os.path.join(stats_dir, mode_name)
            os.makedirs(mode_dir, exist_ok=True)
            
            # Save JSON stats
            json_path = os.path.join(mode_dir, f"style_stats_{domain_name}_{mode_name}.json")
            extractor.save_style_stats_to_json(json_path)
            
            # Save PyTorch stats
            pth_path = os.path.join(mode_dir, f"style_stats_{domain_name}_{mode_name}.pth")
            extractor.save_style_stats(pth_path)
            
            print(f"Saved style stats for {domain_name} (mode {mode_name}) to {json_path}")


    def extract_from_saved_model(
        self, 
        model_path: str,
        domain_name: str,
        model_class: nn.Module,
        model_args: dict,
        results_dir: str = "style_stats",
        domain_indices_to_extract: List[int] = None
    ) -> None:
        """
        Extrahiert Style-Statistiken aus einem gespeicherten Modell-Checkpoint für ALLE Domänen.
        Speichert die Statistiken im JSON-Format pro Extractor-Modus.

        Args:
            model_path: Pfad zum gespeicherten Modell (.pt)
            domain_name: Name der Ziel-Domäne (z.B. "sketch")
            model_class: Klasse des Modells (z.B. resnet50)
            model_args: Argumente zur Modell-Initialisierung
            results_dir: Basis-Verzeichnis für die Speicherung
        """
        # 1. Modell laden
        model = self._load_model(model_path, model_class, model_args)
        model.eval()
        model.enable_style_stats(True)
    
        # 2. Verzeichnis vorbereiten
        #stats_dir = os.path.join(results_dir, f"testtest_{domain_name}")
        #os.makedirs(stats_dir, exist_ok=True)
        """
        domain_idx = self.domain_names.index(domain_name)
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = model(dummy_input, domain_idx=domain_idx)

        """
        print(domain_indices_to_extract)
        if domain_indices_to_extract is None:
            domain_indices_to_extract = list(range(len(self.domain_names)))

        """
        for domain_idx in domain_indices_to_extract:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                # Forward-Pass triggert _update_style_stats() im Modell
                _ = model(dummy_input, domain_idx=domain_idx)
                print(f"Stats for domain {domain_idx}:")
                print("Count:", model.style_stats.count[domain_idx].item())
                print("Layer counts:", model.style_stats.layer_counts[domain_idx])
        """
        

            # Für jeden Extractor-Modus speichern
        for extractor_name, extractor in self.extractors.items():
            #target_layer = extractor.target_layer
            print("Verfügbare Layer in mu_dict:", list(model.style_stats.mu_dict.keys()))
            print("model.style_stats.target_layer:", model.style_stats.target_layer)
            #print("Aktueller target_layer:", target_layer)
            print("\nProcessing extractor:", extractor_name)
            print("Available layers in source:", list(model.style_stats.mu_dict.keys()))

            for domain_idx in domain_indices_to_extract:
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    # Forward-Pass triggert _update_style_stats() im Modell
                    _ = model(dummy_input, domain_idx=domain_idx)
                    print(f"Stats for domain {domain_idx}:")
                    print("Count:", model.style_stats.count[domain_idx].item())
                    print("Layer counts:", model.style_stats.layer_counts[domain_idx])

                # Hole Statistiken aus dem Original-Modell
            if extractor.mode == "single":
                target_layer = int(extractor_name.split('_')[-1])
                #model.style_stats.target_layer = target_layer
                if str(target_layer) in model.style_stats.mu_dict:
                    self._transfer_layer_stats(
                        model.style_stats,
                        extractor,
                        domain_idx,
                        target_layer
                    )
                
            elif extractor.mode == "selective":
                #target_layer =  #hier müssen die jeweiligen layer der config irgendwie rein
                for target_layer in extractor.target_layer:
                    if str(target_layer) in model.style_stats.mu_dict:
                        self._transfer_layer_stats(
                            model.style_stats,
                            extractor,
                            domain_idx,
                            target_layer
                        )
                
            elif extractor.mode == "average":
                for target_layer in range(4):  # All 4 layers
                    if str(target_layer) in model.style_stats.mu_dict:
                        self._transfer_layer_stats(
                            model.style_stats,
                            extractor,
                            domain_idx,
                            target_layer
                        )
                                        
        self.save_style_stats(domain_name, results_dir)


    def _transfer_layer_stats(
        self,
        source_stats: StyleStatistics,
        target_extractor: StyleStatistics,
        domain_idx: int,
        layer_idx: int
    ) -> None:
        """Transfers statistics for a single layer from source to target extractor"""
        layer_key = str(layer_idx)

        if layer_key not in source_stats.mu_dict:
            print(f"Warning: Layer {layer_idx} not found in source stats")
            return
        
        if layer_key not in target_extractor.mu_dict:
            num_channels = source_stats.mu_dict[layer_key].shape[1]
            print(f"Initializing layer {layer_idx} with {num_channels} channels")
            target_extractor._init_layer(layer_idx, num_channels)

        print(f"\n=== Transferring layer stats for {layer_idx} ===")
        print(f"Source mu shape: {source_stats.mu_dict[str(layer_idx)].shape if str(layer_idx) in source_stats.mu_dict else 'N/A'}")
        print(f"Target mu shape: {target_extractor.mu_dict[str(layer_idx)].shape if str(layer_idx) in target_extractor.mu_dict else 'N/A'}")
        print("Original model layers:", list(source_stats.mu_dict.keys()))
        print("Target extractor layers before transfer:", list(target_extractor.mu_dict.keys()))
        print(f"Transferring layer {layer_idx} for domain {domain_idx}")

        src_mu = source_stats.mu_dict[layer_key][domain_idx]
        src_sig = source_stats.sig_dict[layer_key][domain_idx]
    
        if target_extractor.target_layer is None:
            target_extractor.target_layer = layer_idx
        
        # Initialize layer if needed
        if layer_key not in target_extractor.mu_dict:
            target_extractor._init_layer(layer_idx, src_mu.shape[0])
    
        # Transfer stats
        target_extractor.mu_dict[str(layer_idx)].data[domain_idx] = src_mu.clone()
        target_extractor.sig_dict[str(layer_idx)].data[domain_idx] = src_sig.clone()
        target_extractor.count[domain_idx] = source_stats.count[domain_idx]
        target_extractor.layer_counts[domain_idx, layer_idx] = source_stats.layer_counts[domain_idx, layer_idx]
        """

        layer_key = str(layer_idx)
        if layer_key not in source_stats.mu_dict:
            print(f"Warning: Layer {layer_idx} not found in source stats")
            return
        
        num_channels = source_stats.mu_dict[layer_key].shape[1]

        print("\n=== Transferring layer stats ===")
        print(f"Domain: {domain_idx} ({self.domain_names[domain_idx]})")
        print(f"Layer: {layer_idx}")
        print(f"Source layers: {list(source_stats.mu_dict.keys())}")
        print(f"Target layers before: {list(target_extractor.mu_dict.keys())}")

        if layer_key not in target_extractor.mu_dict:
            print(f"Initializing new layer {layer_idx} in target extractor")
            target_extractor._init_layer(layer_idx, num_channels)
        elif target_extractor.mu_dict[layer_key].shape[1] != num_channels:
            print(f"Warning: Channel mismatch for layer {layer_idx} - reinitializing")
            target_extractor._init_layer(layer_idx, num_channels)

        try:
            target_extractor.mu_dict[layer_key].data[domain_idx] = \
            source_stats.mu_dict[layer_key][domain_idx].clone()
            target_extractor.sig_dict[layer_key].data[domain_idx] = \
            source_stats.sig_dict[layer_key][domain_idx].clone()

            target_extractor.count[domain_idx] = source_stats.count[domain_idx]
            target_extractor.layer_counts[domain_idx, layer_idx] = \
            source_stats.layer_counts[domain_idx, layer_idx]

            print("Transfer successful")

        except Exception as e:
            print(f"Error transferring stats for layer {layer_idx}: {str(e)}")
        """


    def _load_model(self, model_path: str, model_class: nn.Module, model_args: dict) -> nn.Module:
        """Loads a model from checkpoint including style statistics
    
        Args:
            model_path: Path to saved model checkpoint
            model_class: Model class to instantiate
            model_args: Arguments for model initialization
        
        Returns:
            Loaded model with restored style statistics
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
        # Load checkpoint with weights_only=True for security
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        print("Keys in checkpoint:", checkpoint.keys())
        print("Style stats keys:", checkpoint["style_stats"].keys())
        print("style stats mu_dict.0:", checkpoint["style_stats"]["mu_dict.0"])
        print("style stats mu_dict.1:", checkpoint["style_stats"]["mu_dict.1"])
        print("style stats mu_dict.2:", checkpoint["style_stats"]["mu_dict.2"])
        print("style stats mu_dict.3:", checkpoint["style_stats"]["mu_dict.3"])
    
        # Initialize model
        model = model_class(**model_args).to(self.device)
    
        # Load model weights (skip style_stats keys)

        #model_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
         #                  if not k.startswith('style_stats.'),}
        #model.load_state_dict(model_state_dict, strict=False)
        model.load_state_dict({
            k: v for k,v in checkpoint['model_state_dict'].items() 
            if not k.startswith('style_stats.')
        }, strict=False)
        print(f"model.style_stats: {model.style_stats}")
    
        # Load style statistics if available
        if 'style_stats' in checkpoint:
            # Ensure the buffers exist
            if not hasattr(model.style_stats, 'layer_counts'):
                print(f"Warning: no model.style_stats.layer_counts, initializing with 0s")
                model.style_stats.register_buffer('layer_counts',
                                                torch.zeros(model.style_stats.num_domains, 
                                                           model.style_stats.num_layers,
                                                           dtype=torch.long))
            if not hasattr(model.style_stats, 'count'):
                print(f"Warning: no model.style_stats.count, initializing with 0s")
                model.style_stats.register_buffer('count',
                                                torch.zeros(model.style_stats.num_domains,
                                                          dtype=torch.long))
        
            
            for key, value in checkpoint['style_stats'].items():
                if key.startswith('mu_dict'):
                    layer = key.split('.')[-1]
                    if f'mu_dict.{layer}' not in model.style_stats.mu_dict:
                        print(f"initializing model.style_stats.mu_dict for layer {layer}")
                        model.style_stats._init_layer(int(layer), value.shape[1])
            
                elif key.startswith('sig_dict'):
                    layer = key.split('.')[-1]
                    if f'sig_dict.{layer}' not in model.style_stats.sig_dict:
                        model.style_stats._init_layer(int(layer), value.shape[1])
            
            # Load the style stats
            model.style_stats.load_state_dict(checkpoint['style_stats'], strict=False)
    
        # Enable style stats collection
        model.enable_style_stats(True)
    
        return model