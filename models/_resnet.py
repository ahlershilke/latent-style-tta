import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Type
from torch.utils import model_zoo
from ._mixstyle import MixStyle
from ._styleextraction import StyleStatistics, StyleExtractorManager
from typing import Tuple


"""
Code taken and slightly adapted from: 
 https://github.com/VoErik/domain-generalization/blob/main/domgen/model_training/_resnet.py
"""


model_urls = {
    'resnet18':
        'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
        'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
        'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
        'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
        'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(nn.Module):
    """
    ResNet model with integrated MixStyle as proposed in Zhou et al. (2021): https://arxiv.org/abs/2104.02008
    """
    def __init__(
            self,
            block: Type[nn.Module],
            layers: list,
            num_classes: int,
            num_domains: int,
            domain_names: List[str] = None,
            fc_dims=None,
            dropout_p=None,
            style_stats_config: dict = None,
            use_mixstyle=True,
            mixstyle_layers: list = [],
            mixstyle_p: float = 0.5,
            mixstyle_alpha: float = 0.3,
            verbose: bool = False,
            **kwargs
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.feature_dim = 512 * block.expansion
        self.num_domains = num_domains
        self.domain_names = domain_names

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(
            fc_dims, 512 * block.expansion, dropout_p
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.style_stats_enabled = True
        self.style_stats_config = style_stats_config or {
            'mode': 'selective',
            'target_layer': [0,1,2,3],
            'use_ema': True,
            'ema_momentum': 0.9
        }

        self.style_stats = StyleStatistics(
            num_domains=num_domains,
            num_layers=4,
            domain_names=self.domain_names,
            mode=self.style_stats_config.get('mode', 'single'),
            layer_config=self.style_stats_config.get('target_layer', 0),
            use_ema=self.style_stats_config.get('use_ema', True),
            ema_momentum=self.style_stats_config.get('ema_momentum', 0.9),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.style_manager = StyleExtractorManager(
            domain_names=[str(i) for i in range(num_domains)],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.mixstyle = None
        if use_mixstyle:
            self.mixstyle = MixStyle(
                p=mixstyle_p, 
                alpha=mixstyle_alpha, 
                mix='random', 
                num_domains=num_domains,
                num_layers=len(mixstyle_layers),
                store_stats=True
            )
            if verbose:
                print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers

        self._hook_handles = []
        self._feature_maps = {}
        self._register_hooks()
        

    def _register_hooks(self):
        self._remove_hooks()
        self._feature_maps = {} #reset bei jedem forward pass #TODO will man das???

        layers = [
            ('layer1', self.layer1[-1]),
            ('layer2', self.layer2[-1]), 
            ('layer3', self.layer3[-1]),
            ('layer4', self.layer4[-1])
        ]

        for name, layer in layers:
            def hook_factory(layer_name):
                def hook(_, __, output):
                    if not self.training:
                        return
                    self._feature_maps[layer_name] = output.detach()
                return hook

            handle = layer.register_forward_hook(hook_factory(name))
            self._hook_handles.append(handle)


    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._feature_maps = {}


    def enable_style_stats(self, enable=True):
        #Aktiviert/deaktiviert das Sammeln von Style Statistiken
        self.style_stats_enabled = enable


    def _make_layer(
            self,
            block: Type[nn.Module],
            out_channels: int,
            num_blocks: int,
            stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            downsample = None
            if stride != 1 or self.in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _construct_fc_layer(
            self,
            fc_dims: List[int],
            input_dim: int,
            dropout_p: float = None
    ) -> nn.Sequential|None:
        """
        Constructs fully connected layer

        :param fc_dims: dimensions of fc layers, if None, no fc layers are constructed
        :param input_dim: input dimension
        :param dropout_p: dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _feature_extractor(
            self,
            x: Tensor,
            domain_idx: int = None
    ) -> Tensor:
        """Pass through the ResNet-blocks."""

        self._feature_maps = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #layer_outputs = []

        x = self.layer1(x)
        #layer_outputs.append(x)
        if self.mixstyle is not None and 'layer1' in self.mixstyle_layers:
            x = self.mixstyle(x)
        #self._update_style_stats(x, domain_idx, layer_idx=0)
        #if self.mixstyle is not None and 'layer1' in self.mixstyle_layers:
         #   x = self.mixstyle(x, domain_labels=domain_idx, layer_idx=0)

        x = self.layer2(x)
        #layer_outputs.append(x)
        if self.mixstyle is not None and 'layer2' in self.mixstyle_layers:
            x = self.mixstyle(x)
        #self._update_style_stats(x, domain_idx, layer_idx=1)
        #if self.mixstyle is not None and 'layer2' in self.mixstyle_layers:
         #   x = self.mixstyle(x, domain_labels=domain_idx, layer_idx=1)

        x = self.layer3(x)
        #layer_outputs.append(x)
        if self.mixstyle is not None and 'layer3' in self.mixstyle_layers:
            x = self.mixstyle(x)
        #self._update_style_stats(x, domain_idx, layer_idx=2)
        #if self.mixstyle is not None and 'layer3' in self.mixstyle_layers:
         #   x = self.mixstyle(x, domain_labels=domain_idx, layer_idx=2)

        x = self.layer4(x)
        #layer_outputs.append(x)
        if self.mixstyle is not None and 'layer4' in self.mixstyle_layers:
            x = self.mixstyle(x)
        #self._update_style_stats(x, domain_idx, layer_idx=3) # noch nötig für Style Transfer?
        #if self.mixstyle is not None and 'layer4' in self.mixstyle_layers:
         #   x = self.mixstyle(x, domain_labels=domain_idx, layer_idx=3)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)

        #if domain_idx is not None:
         #   self._update_style_stats(domain_idx)

        #return x, layer_outputs
        layer_outputs = [self._feature_maps.get(f'layer{i+1}') for i in range(4)]
        if domain_idx is not None and self.training:
            for i, features in enumerate(layer_outputs):
                if features is not None:
                    self._update_style_stats(x=features, domain_idx=domain_idx, layer_idx=i)
        """
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)
        """
        return x, layer_outputs

    def forward(
            self,
            x: Tensor,
            domain_idx: int = None
    ) -> Tensor:
        out, layer_outputs = self._feature_extractor(x, domain_idx)

        """
        if self.training and domain_idx is not None:
            for layer_idx in range(4):  # Für alle relevanten Layers
                self._update_style_stats(out, domain_idx, layer_idx)
        
        if domain_idx is not None: #and self.training
            for layer_idx, features in enumerate(layer_outputs):
                self._update_style_stats(features, domain_idx, layer_idx)
        """
        out = self.avgpool(out)
        v = out.view(out.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        y = self.classifier(v)
        return y
    
    #TODO ??
    def get_style_stats(self, domain_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        #Returns the style statistics of the model.
        return self.style_stats.get_style_stats(domain_idx)
    
    def _update_style_stats(self, x: torch.Tensor, domain_idx: torch.Tensor, layer_idx: int):
        #Collects μ and σ for layers and domains.
        if not (self.style_stats_enabled and self.training and domain_idx is not None):
            return
        
        """
        for layer_name, features in self._feature_maps.items():
            if x.dim() != 4:
                continue

            layer_idx = int(layer_name[-1]) - 1
        
            #print(f"\nUpdating stats for Layer {layer_idx} | Input shape: {x.shape}")

            mu = x.mean(dim=[2, 3], keepdim=True) # True)  # [B, C, 1, 1]
            sig = x.std(dim=[2, 3], keepdim=True) #True)
        
            #print(f"Layer {layer_idx} - mu: {mu.min().item():.4f}-{mu.max().item():.4f} | "
             #   f"sig: {sig.min().item():.4f}-{sig.max().item():.4f}")

            if str(layer_idx) not in self.style_stats.mu_dict: #or
                #self.style_stats.mu_dict[str(layer_idx)].shape[1] != mu.shape[1]):  # Channel-Dimension prüfen
                self.style_stats._init_layer(layer_idx, mu.shape[1])

            #if str(layer_idx) in self.style_stats.mu_dict:
                #print(f"Existing stats - mu: {self.style_stats.mu_dict[str(layer_idx)].min().item():.4f}-"
                 #   f"{self.style_stats.mu_dict[str(layer_idx)].max().item():.4f}")
        
            if isinstance(domain_idx, int):
                domain_idx = torch.tensor([domain_idx], device=x.device)
                #elf.style_stats._update(domain_idx, layer_idx, mu, sig)
        
            self.style_stats._update(domain_idx, layer_idx, mu, sig)
            #print(f"Updated stats - mu: {self.style_stats.mu_dict[str(layer_idx)][domain_idx].min().item():.4f}-"
             #   f"{self.style_stats.mu_dict[str(layer_idx)][domain_idx].max().item():.4f}")
            #self.style_stats._batch_update(domain_idx, layer_idx, mu, sig)
        """
        if x.dim() != 4:
            return

        mu = x.mean(dim=[2, 3], keepdim=True).detach()
        sig = x.std(dim=[2, 3], keepdim=True).detach()

        # Initialisiere Layer falls nötig
        if str(layer_idx) not in self.style_stats.mu_dict:
            self.style_stats._init_layer(layer_idx, mu.shape[1])

        # Konvertiere Domain-Indizes falls nötig
        if isinstance(domain_idx, int):
            domain_idx = torch.tensor([domain_idx], device=x.device)
    
        """
        # Update für jede Domain im Batch
        for domain in domain_idx.unique():
            mask = (domain_idx == domain)
            self.style_stats._batch_update(
                domain.item(),  # Einzelner Index
                layer_idx,
                mu[mask],      # Nur Features dieser Domain
                sig[mask]
            )
        """
        self.style_stats._batch_update(domain_idx, layer_idx, mu, sig)

    
    #TODO ersatz für _update_style_stats() ?
    def _update_stats(self, features, domain_idx, layer_idx):
        mu = features.mean(dim=[2, 3], keepdim=True).detach()  # [B, C, 1, 1]
        sigma = features.std(dim=[2, 3], keepdim=True).detach()
        self.style_stats._update(domain_idx, layer_idx, mu, sigma)


    def save_style_stats(self, path: str, mode: str = 'all'):
        """
        Speichert Style-Statistiken
        :param path: Pfad zum Speichern
        :param mode: 'all' für alle Modi oder spezifischer Modus
        """
        if mode == 'all':
            self.style_manager.save_style_stats("all_domains", path)
        else:
            torch.save({
                'model_state': self.state_dict(),
                'style_stats': self.style_stats.state_dict(),
                'config': {
                    'style_stats_config': self.style_stats_config,
                    'target_layer': self.style_stats.target_layer,
                    'num_domains': self.style_stats.num_domains
                    }
            }, path)

    @classmethod
    def load_with_style_stats(cls, path: str, **model_args):
        """
        Lädt Modell mit gespeicherten Style-Statistiken
        :param path: Pfad zur gespeicherten Datei
        :param model_args: Argumente für Modellinitialisierung
        """
        checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model = cls(**model_args)
        
        """
        model_state_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                      if k in model_state_dict}

        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)
        
        if 'style_stats' in checkpoint:
            model.style_stats.load_state_dict(checkpoint['style_stats'], strict=False)
            model.style_stats_config = checkpoint.get('style_stats_config', {})
        """

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
        if not hasattr(model.style_stats, 'layer_counts'):
            model.style_stats.register_buffer('layer_counts',
                                              torch.zeros(model.style_stats.num_domains, model.style_stats.num_layers, dtype=torch.long))
        if not hasattr(model.style_stats, 'count'):
            model.style_stats.register_buffer('count',
                                              torch.zeros(model.style_stats.num_domains, dtype=torch.long))
    
        # Load style stats if available
        if 'style_stats' in checkpoint:
            model.style_stats.load_state_dict(checkpoint['style_stats'], strict=False)
        
        return model    
    

class BasicBlock (nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: 1,
            downsample=None
    ):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # skip layer
        out = self.relu(out)

        return out


class Bottleneck (nn.Module):
    expansion = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample=None
    ):
        super(Bottleneck, self).__init__()
        # 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution to process spatial dimensions
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution to expand channels back
        self.conv3 = nn.Conv2d(
            out_channels,
            self.expansion * out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # skip layer
        out = self.relu(out)

        return out


def init_pretrained_weights(model, model_url, verbose=False):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    if verbose:
        print("Initializing pretrained weights...")
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def resnet18(num_classes: int, num_domains: int, verbose=False, **kwargs):
    model = ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes, num_domains, verbose=verbose, **kwargs
    )
    if kwargs.get('pretrained', False):
        init_pretrained_weights(model, model_urls['resnet18'], verbose=verbose)
    return model


def resnet34(num_classes: int, num_domains: int, pretrained: bool = True, verbose=False, **kwargs):
    model = ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes, num_domains, verbose=verbose, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'], verbose=verbose)
    return model


def resnet50(num_classes: int, num_domains: int, pretrained: bool = True, verbose=False, **kwargs):
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes, num_domains, verbose=verbose, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'], verbose=verbose)
    return model


def resnet101(num_classes: int, num_domains: int, pretrained: bool = True, verbose=False, **kwargs):
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], num_classes, num_domains, verbose=verbose, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'], verbose=verbose)
    return model


def resnet152(num_classes: int, num_domains: int, pretrained: bool = True, verbose=False, **kwargs):
    model = ResNet(
        Bottleneck, [3, 8, 36, 3], num_classes, num_domains, verbose=verbose, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'], verbose=verbose)
    return model