import torch
import torch.nn as nn
import json


class StyleStatistics(nn.Module):
    def __init__(self, num_domains: int, num_layers: int):
        super().__init__()
        self.num_domains = num_domains
        self.num_layers = num_layers

        # initialising tensors for mu/sig per layer & domain
        self.register_buffer("mu", torch.zeros(num_layers, num_domains, 1, 1, 1))  # shape: [layers, domains, 1, 1, 1]
        self.register_buffer("sig", torch.zeros(num_layers, num_domains, 1, 1, 1))
        self.register_buffer("count", torch.zeros(num_domains, dtype=torch.long))  # sample counter per domain


    def update(self, domain_idx: int, layer_idx: int, mu: torch.Tensor, sig: torch.Tensor):
        """
        Updates the running average of the mean and standard deviation for the given domain and layer.
        Args:
            domain_idx: index of the domain (0 to num_domains-1)
            layer_idx: index of layer (0 to num_layers-1)
            mu: mean of current batch (shape: [B, C, 1, 1])
            sig: standard deviation of current batch (shape: [B, C, 1, 1])
        """
        # average batch statistics (over batch dimension B)
        mu_batch_avg = mu.mean(dim=0, keepdim=True)  # shape: [1, C, 1, 1]
        sig_batch_avg = sig.mean(dim=0, keepdim=True)

        # exponential moving average
        # to make it more robust against domain inbalance
        if self.count[domain_idx] == 0:
            self.mu[layer_idx, domain_idx] = mu_batch_avg
            self.sig[layer_idx, domain_idx] = sig_batch_avg
        else:
            self.mu[layer_idx, domain_idx] = (self.mu[layer_idx, domain_idx] * self.count[domain_idx] + mu_batch_avg) / (self.count[domain_idx] + 1)
            self.sig[layer_idx, domain_idx] = (self.sig[layer_idx, domain_idx] * self.count[domain_idx] + sig_batch_avg) / (self.count[domain_idx] + 1)
        
        self.count[domain_idx] += 1

  
    def save_style_stats_to_json(self, filepath: str):
        """
        Saves the StyleStatistics to a JSON file.
        Args:
            filepath: Path to save the JSON file.
        """
        stats_dict = {
            "mu": self.mu.cpu().tolist(),
            "sig": self.sig.cpu().tolist(),
            "count": self.count.cpu().tolist(),
            "num_layers": self.num_layers,
            "num_domains": self.num_domains,
        }

        with open(filepath, "w") as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Style statistics saved to: {filepath}")


    def load_style_stats_from_json(filepath: str):
        """
        Loads the StyleStatistics from a JSON file.
        Args:
            filepath: Path to the JSON file.
        """
        with open(filepath, "r") as f:
            stats_dict = json.load(f)

        num_layers = stats_dict["num_layers"]
        num_domains = stats_dict["num_domains"]
        style_stats = StyleStatistics(num_domains=num_domains, num_layers=num_layers)

        style_stats.mu = torch.tensor(stats_dict["mu"])
        style_stats.sig = torch.tensor(stats_dict["sig"])
        style_stats.count = torch.tensor(stats_dict["count"], dtype=torch.long)

        return style_stats
    

    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path, device="cuda"):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)


