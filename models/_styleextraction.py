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
        mode: str = "single",  # "average", "selective", "paired", "single"
        layer_config=None,      # for "selective", "paired" or "single"
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

        #TODO which one is it??
        self.register_buffer('layer_counts', 
                             torch.zeros(num_domains, num_layers, dtype=torch.long))
        self.register_buffer('count', torch.zeros(num_domains, dtype=torch.long))

        # initialize storage structures
        self.mu_dict = nn.ParameterDict()  # {layer_idx: parameter[num_domains, C, 1, 1]}
        self.sig_dict = nn.ParameterDict()
        #self.register_buffer('count', torch.zeros(num_domains, dtype=torch.long))

        if self.mode == "selective":
            if not layer_config:
                raise ValueError("layer_config required for selective mode")
            self.target_layers = layer_config
            self.layer_momentum = nn.ParameterDict({
                str(layer): nn.Parameter(torch.tensor(ema_momentum))
                for layer in self.target_layers
            })
        elif self.mode == "paired":
            if not layer_config:
                raise ValueError("layer_config required for paired mode")
            self.layer_pairs = layer_config
        elif self.mode == "single":
            self.target_layer = kwargs.get("target_layer", 0)
        elif self.mode == "attention":
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        else:
            if mode not in ["average", "all"]:
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
        assert mu.shape[0] == domain_idx.shape[0], "Batch dimension mismatch"
    
        # Squeeze auf [B,C] falls nötig
        mu = mu.squeeze(-1).squeeze(-1) if mu.dim() == 4 else mu
        sig = sig.squeeze(-1).squeeze(-1) if sig.dim() == 4 else sig
    
        for d in torch.unique(domain_idx):
            mask = (domain_idx == d)
            if mask.sum() == 0:
                continue
            
            # Vektorisiertes Mittelwertberechnung
            #mu_mean = mu[mask].mean(dim=0)  # [C,]
            #sig_mean = sig[mask].mean(dim=0) # [C,]
        
            # EMA Update mit dynamischem Momentum
            self._ema_update(d.item(), layer_idx, mu[mask], sig[mask])


    def _update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        """Update statistics for given domain and layer with mode-specific rules"""
        if not self._should_update_layer(layer_idx):
            return

        #mu = mu.squeeze(-1).squeeze(-1)  # [B, C]
        #sig = sig.squeeze(-1).squeeze(-1)  # [B, C]

        #B = mu.shape[0]

        if not isinstance(domain_idx, torch.Tensor):
            #domain_idx = torch.tensor(domain_idx, device=mu.device)
            domain_idx = torch.tensor([domain_idx], device=mu.device)

        self._batch_update(domain_idx, layer_idx, mu, sig)
        

    def _should_update_layer(self, layer_idx: int) -> bool:
        """Check if layer should be updated based on current mode."""
        if self.mode == "selective":
            return layer_idx in self.target_layers
        elif self.mode == "paired":
            return any(layer_idx in pair for pair in self.layer_pairs)
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
    
    """
    def _ema_update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        #Perform EMA update with mode-specific momentum.
        self._validate_inputs(domain_idx, mu, sig)
        
        #TODO layer_counts existiert bisher nicht
        update_count = self.layer_counts[domain_idx, layer_idx]
        momentum = self._get_momentum(layer_idx, update_count)
    
        if str(layer_idx) not in self.mu_dict:
            self._init_layer(layer_idx, mu.size(-1))

        mu_mean = mu.mean(dim=0)
        sig_mean = sig.mean(dim=0)
    
        if self.count[domain_idx] == 0:
            self.mu_dict[str(layer_idx)][domain_idx] = mu_mean
            self.sig_dict[str(layer_idx)][domain_idx] = sig_mean
        else:
            self.mu_dict[str(layer_idx)][domain_idx] = (
                momentum * self.mu_dict[str(layer_idx)][domain_idx] + 
                (1 - momentum) * mu_mean
            )   
            self.sig_dict[str(layer_idx)][domain_idx] = (
                momentum * self.sig_dict[str(layer_idx)][domain_idx] +
                (1 - momentum) * sig_mean
            )
            
        self.count[domain_idx] += 1
        self.layer_counts[domain_idx, layer_idx] += 1
    
    
    def _ema_update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        #Perform EMA update with mode-specific momentum.
        self._validate_inputs(domain_idx, mu, sig)
    
        # Squeeze spatial dimensions (falls [B, C, 1, 1] → [B, C])
        mu = mu.squeeze(-1).squeeze(-1) if mu.dim() == 4 else mu
        sig = sig.squeeze(-1).squeeze(-1) if sig.dim() == 4 else sig
    
        update_count = self.layer_counts[domain_idx, layer_idx]
        momentum = self._get_momentum(layer_idx, update_count)

        # Channel-Dimension extrahieren (z. B. 256, 512, 1024, 2048)
        num_channels = mu.size(1)  # [B, C] → C

        # Initialisiere Layer-Buffer, falls nicht vorhanden oder falsche Dimension
        if str(layer_idx) not in self.mu_dict:
            self._init_layer(layer_idx, num_channels)
        elif self.mu_dict[str(layer_idx)].shape[1] != num_channels:
            # Falls Channel-Dimension nicht übereinstimmt, neu initialisieren
            self._init_layer(layer_idx, num_channels)

        # Berechne Mittelwert über Batch-Dimension (dim=0)
        mu_mean = mu.mean(dim=0)  # [C]
        sig_mean = sig.mean(dim=0)  # [C]

        if self.count[domain_idx] == 0:
            # Erster Update: Direkte Zuweisung
            self.mu_dict[str(layer_idx)].data[domain_idx] = mu_mean
            self.sig_dict[str(layer_idx)].data[domain_idx] = sig_mean
        else:
            # EMA-Update
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
    """

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


    # TODO funktionalität ist noch nicht ganz da, man kann nicht richtig zwischen modes switchen
    def get_style_stats(self, domain_idx: int):
        """Returns the style statistics for the given domain index.
    
        Args:
            domain_idx: Index of the target domain.
        
        Returns:
            Tuple of (mu, sig) where:
                - mu: Combined mean statistics
                - sig: Combined standard deviation statistics
        """
        
        if self.mode == "single":
            # statistics from a single target layer
            mu = self.mu_dict[str(self.target_layer)][domain_idx]
            sig = self.sig_dict[str(self.target_layer)][domain_idx]
        
        elif self.mode == "average":
            # averaging across all layers
            mu = sum(self.mu_dict[layer][domain_idx] for layer in self.mu_dict)
            sig = sum(self.sig_dict[layer][domain_idx] for layer in self.sig_dict)
        
        elif self.mode == "selective":
            # averaging across selected layers
            mu = sum(self.mu_dict[str(layer)][domain_idx] for layer in self.target_layers)
            sig = sum(self.sig_dict[str(layer)][domain_idx] for layer in self.target_layers)
        
        elif self.mode == "paired":
            # pairwise averaging
            mu = sum(
                self.mu_dict[str(pair[0])][domain_idx] + self.mu_dict[str(pair[1])][domain_idx]
                for pair in self.layer_pairs
            )
            sig = sum(
                self.sig_dict[str(pair[0])][domain_idx] + self.sig_dict[str(pair[1])][domain_idx]
                for pair in self.layer_pairs
            )
        
        """
        elif self.mode == "all":
            # averaging across all layers, similar to "average" but with explicit layer handling
            mu = sum(self.mu_dict[str(i)][domain_idx] for i in range(self.num_layers))
            sig = sum(self.sig_dict[str(i)][domain_idx] for i in range(self.num_layers))
        
        
        elif self.mode == "attention":
            # weighted average based on layer weights
            assert len(self.layer_weights) == len(self.mu_dict), "Layer weights mismatch!"
    
            weights = torch.softmax(self.layer_weights, dim=0)
    
            mu = sum(
                weights[i] * self.mu_dict[str(i)][domain_idx] 
                for i in range(len(self.mu_dict))
            )
            sig = sum(
                weights[i] * self.sig_dict[str(i)][domain_idx] 
                for i in range(len(self.sig_dict))
            )
        """

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
            return [float(x) for x in tensor.squeeze().cpu().numpy().ravel()]

        stats_dict = {
            "domain_stats": {
                str(domain_id): {
                    "layers": [
                        {
                            "name": f"layer_{layer_idx}",
                            "mu": flatten_stats(self.mu_dict[str(layer_idx)][domain_id]),
                            "sig": flatten_stats(self.sig_dict[str(layer_idx)][domain_id])
                        }
                        for layer_idx in sorted(self.mu_dict.keys())
                    ],
                    "count": int(self.count[domain_id].item())
                }
                for domain_id in range(self.num_domains)
            },
            "metadata": {
                "num_layers": self.num_layers,
                "mode": self.mode
            }
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
            num_domains=stats_dict["num_domains"],
            num_layers=stats_dict["num_layers"],
            mode=stats_dict.get("mode", "average")  # default to "average" if not present
        )

        # load mu and sig from nested dictionary structure
        for layer_str, layer_data in stats_dict["mu"].items():
            style_stats.mu_dict[layer_str] = nn.Parameter(
                torch.tensor(layer_data),
                requires_grad=False
            )
    
        for layer_str, layer_data in stats_dict["sig"].items():
            style_stats.sig_dict[layer_str] = nn.Parameter(
                torch.tensor(layer_data),
                requires_grad=False
            )

        style_stats.count = torch.tensor(stats_dict["count"], dtype=torch.long)

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
            'selective_0_1': {'mode': 'selective', 'config': [0, 1]},
            'selective_0_2': {'mode': 'selective', 'config': [0, 2]},
            'selective_0_3': {'mode': 'selective', 'config': [0, 3]},
            'selective_1_2': {'mode': 'selective', 'config': [1, 2]},
            'selective_1_3': {'mode': 'selective', 'config': [1, 3]},
            'selective_2_3': {'mode': 'selective', 'config': [2, 3]},
            'selective_0_1_2': {'mode': 'selective', 'config': [0, 1, 2]},
            'selective_0_1_3': {'mode': 'selective', 'config': [0, 1, 3]},
            'selective_1_2_3': {'mode': 'selective', 'config': [1, 2, 3]},
            'paired': {'mode': 'paired', 'config': [(0,1), (2,3)]},
            'average': {'mode': 'average', 'config': None}
        }
    

    def _create_style_extractors(self) -> Dict[str, StyleStatistics]:
        """Creates StyleExtractor instances for all desired modes"""
        modes = self._get_default_modes()
        extractors = {}
        
        for name, params in modes.items():
            extractors[name] = StyleStatistics(
                num_domains=len(self.domain_names),
                num_layers=4,
                mode=params['mode'],
                layer_config=params['config'],
                device=self.device
            )
        
        return extractors
    

    def save_style_stats(self, domain_name: str, results_dir: str = "style_stats") -> None:
        """Saves all collected style statistics"""
        stats_dir = os.path.join(results_dir, f"test_{domain_name}")
        os.makedirs(stats_dir, exist_ok=True)
    
        for name, extractor in self.extractors.items():
            json_stats_path = os.path.join(stats_dir, f"style_stats_{name}.json")
            pth_stats_path = os.path.join(stats_dir, f"style_stats_{name}.pth")
            extractor.save_style_stats_to_json(json_stats_path)
            extractor.save_style_stats(pth_stats_path)
            print(f"Saved style stats {name} for {domain_name} to {json_stats_path}")
    
    """
    def extract_and_save_style_stats(self, model: nn.Module, domain_name: str, 
                                   results_dir: str = "style_stats") -> None:
        
        Extracts style statistics from a trained model using all available modes
        
        Args:
            model: Trained model to extract style statistics from
            domain_name: Name of the test domain (e.g., 'sketch', 'photo')
            results_dir: Root directory for saving style statistics
        
        # Create directory structure
        stats_dir = os.path.join(results_dir, f"test_{domain_name}")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Update stats for all extractors
        for name, extractor in self.extractors.items():
            # Reset extractor for fresh stats
            extractor.reset_stats()
            
            # Update stats for all domains
            for domain_idx in range(len(self.domain_names)):
                mu, sig = model.get_style_stats(domain_idx)
                extractor.set_stats_for_domain(domain_idx, mu, sig)
            
            # Save stats
            stats_path = os.path.join(stats_dir, f"style_stats_{name}.json")
            extractor.save_style_stats_to_json(stats_path)
            print(f"Saved {name} stats for {domain_name} to {stats_path}")
    

    
    def extract_from_saved_model(self, model_path: str, domain_name: str, 
                               model_class: nn.Module, model_args: dict,
                               results_dir: str = "style_stats") -> None:
        
        Loads a saved model and extracts style statistics
        
        Args:
            model_path: Path to saved model checkpoint
            domain_name: Name of domain for naming output files
            model_class: Model class to instantiate
            model_args: Arguments needed to initialize the model
            results_dir: Directory to save results
    
        # Load model
        model = self._load_model(model_path, model_class, model_args)
        model.eval()
        
        # Extract and save stats
        self.extract_and_save_style_stats(
            model=model,
            domain_name=domain_name,
            results_dir=results_dir
        )
    """

    def extract_from_saved_model(
        self, 
        model_path: str,
        domain_name: str,
        model_class: nn.Module,
        model_args: dict,
        results_dir: str = "style_stats"
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
    
        # 2. Verzeichnis vorbereiten
        stats_dir = os.path.join(results_dir, f"final_{domain_name}")
        os.makedirs(stats_dir, exist_ok=True)

        # 3. Für jede Domäne Statistiken extrahieren
        for domain_idx in range(len(self.domain_names)):
            # Dummy-Input für Forward-Pass (nötig, um mu/sig zu berechnen)
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
            with torch.no_grad():
                # Forward-Pass triggert _update_style_stats() im Modell
                model(dummy_input, domain_idx=domain_idx)

            # Für jeden Extractor-Modus speichern
            for extractor_name, extractor in self.extractors.items():
                # Hole Statistiken im jeweiligen Modus (single/average/...)
                mu, sig = model.get_style_stats(domain_idx)
            
                # Speichere als JSON
                stats_path = os.path.join(stats_dir, f"{extractor_name}_domain_{domain_idx}.json")
                extractor.save_style_stats_to_json(stats_path)
                print(f"Saved {extractor_name} stats (domain {domain_idx}) to {stats_path}")
    

    def _load_model(self, model_path: str, model_class: nn.Module, model_args: dict) -> nn.Module:
        """Helper to load a model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = model_class(**model_args).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model