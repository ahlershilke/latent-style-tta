import torch
import torch.nn as nn
import json


class StyleStatistics(nn.Module):
    DEFAULT_MOMENTUM = 0.9

    def __init__(self,
        num_domains: int,
        num_layers: int,
        mode: str = "selective",  # "average", "selective", "paired", "attention", "single"
        layer_config=None,      # for "selective", "paired" or "single"
        use_ema: bool = True,  # exponential moving average
        ema_momentum: float = DEFAULT_MOMENTUM,  # momentum for EMA
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
                - "attention": Weighted average using learned attention weights.
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

        # initialize storage structures
        self.mu_dict = nn.ParameterDict()  # {layer_idx: parameter[num_domains, C, 1, 1]}
        self.sig_dict = nn.ParameterDict()
        self.register_buffer('count', torch.zeros(num_domains, dtype=torch.long))

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
            if mode not in ["average", "attention"]:
                raise ValueError(f"Unknown mode: {mode}")

        # initialize buffers for first use tracking
        self._initialized = True
    
    
    def _initialize_buffers(self):
        """Initializes the data structures for storing style statistics."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mu_dict = nn.ParameterDict()
        self.sig_dict = nn.ParameterDict()
    
        self.register_buffer("count", torch.zeros(self.num_domains, dtype=torch.long, device=device))
        self._initialized = True    # to prevent re-initialization


    def _update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        """Update statistics for given domain and layer with mode-specific rules"""
        if not self._should_update_layer(layer_idx):
            return
    
        mu = mu.squeeze(-1).squeeze(-1)  # shape: [B, C]
        sig = sig.squeeze(-1).squeeze(-1)  # shape: [B, C]
    
        for d in domain_idx.unique():
            mask = domain_idx == d
            mu_mean = mu[mask].mean(dim=0)
            sig_mean = sig[mask].mean(dim=0)
        
            if self.use_ema:
                self._ema_update(d.item(), layer_idx, mu_mean, sig_mean)
            else:
                self._simple_update(d.item(), layer_idx, mu_mean, sig_mean)


    def _should_update_layer(self, layer_idx: int) -> bool:
        """Check if layer should be updated based on current mode."""
        if self.mode == "selective":
            return layer_idx in self.target_layers
        elif self.mode == "paired":
            return any(layer_idx in pair for pair in self.layer_pairs)
        elif self.mode == "single":
            return layer_idx == self.target_layer
        
        return True  # for 'average' and 'attention' modes


    def _get_momentum(self, layer_idx: int) -> float:
        """Get momentum value for specific layer and mode."""
        if self.mode == "selective" and str(layer_idx) in self.layer_momentum:
            return torch.sigmoid(self.layer_momentum[str(layer_idx)])
        
        return self.ema_momentum
    

    def _init_layer(self, layer_idx: int, num_channels: int):
        """Initialize buffers for a specific layer"""
        device = self.count.device
    
        self.mu_dict[str(layer_idx)] = nn.Parameter(
            torch.zeros(self.num_domains, num_channels, device=device),
            requires_grad=False
        )
        self.sig_dict[str(layer_idx)] = nn.Parameter(
            torch.zeros(self.num_domains, num_channels, device=device),
            requires_grad=False
        )


    def _ema_update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        """Perform EMA update with mode-specific momentum."""
        momentum = self._get_momentum(layer_idx)
    
        if str(layer_idx) not in self.mu_dict:
            self._init_layer(layer_idx, mu.size(0))
    
        if self.count[domain_idx] == 0:
            self.mu_dict[str(layer_idx)][domain_idx] = mu
            self.sig_dict[str(layer_idx)][domain_idx] = sig
        else:
            self.mu_dict[str(layer_idx)][domain_idx] = (
                momentum * self.mu_dict[str(layer_idx)][domain_idx] + 
                (1 - momentum) * mu
            )   
            self.sig_dict[str(layer_idx)][domain_idx] = (
                momentum * self.sig_dict[str(layer_idx)][domain_idx] +
                (1 - momentum) * sig
            )
            
        self.count[domain_idx] += 1


    def _simple_update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        """Perform simple average update."""
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
    

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)


    @classmethod
    def load(cls, self, path, device="cuda") -> None:
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)


