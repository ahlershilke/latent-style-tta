import torch
import torch.nn as nn

class TTAClassifier(nn.Module):
    def __init__(self, backbone, classifier, domain_shifts, ensemble_mode='mean'):
        """
        backbone: gefrorenes ResNet ohne Klassifikationskopf
        classifier: z. B. ein MLP oder linearer Klassifikator
        domain_shifts: Liste von Vektoren oder nn.Parameter (m bekannte Domänen)
        ensemble_mode: 'mean' oder 'weighted'
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.domain_shifts = domain_shifts  # Liste mit Vektoren, z. B. [torch.Tensor]
        self.ensemble_mode = ensemble_mode

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)  # [B, D, 1, 1]
            feats = feats.view(feats.size(0), -1)  # [B, D]

        outputs = [self.classifier(feats)]  # Original

        for shift in self.domain_shifts:
            aug_feats = feats + shift  # [B, D]
            out = self.classifier(aug_feats)
            outputs.append(out)

        logits = torch.stack(outputs, dim=0)  # [m+1, B, num_classes]

        if self.ensemble_mode == 'mean':
            final_logits = logits.mean(dim=0)
        else:
            # Erweiterbar: z. B. gewichtetes Mittel
            raise NotImplementedError("Only mean ensemble implemented so far.")

        return final_logits
