#!/usr/bin/env python3
"""
Neural Gating Network minimale: impression-level, produce [w_CF, w_CBF] via
softmax a partire da feature pre-click. Se CF_AVAILABLE=0, w_CF viene forzato
a 0 (e w_CBF a 1) DOPO il softmax, tramite masking esplicito e
rinormalizzazione — la rete non puo' in alcun modo assegnare peso effettivo
al CF quando il suo ranking e' degenere, indipendentemente da cosa
produrrebbe il softmax.

Architettura compatta e fissa:
    Input(n_features) -> Dense(hidden[0]) -> ReLU -> [Dense(hidden[1]) -> ReLU] -> Dense(2) -> Softmax
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GatingNet(nn.Module):
    def __init__(self, n_features: int, hidden: List[int]):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, cf_available):
        """x: (B, n_features) float32. cf_available: (B,) 0/1 float32.
        Ritorna w: (B, 2) = [w_CF, w_CBF], con w_CF forzato a 0 dove
        cf_available==0 (e w_CBF=1 li'), mai appreso dal softmax grezzo.
        """
        logits = self.net(x)
        w = torch.softmax(logits, dim=-1)
        w_cf = w[:, 0] * cf_available  # maschera dura: 0 se CF_AVAILABLE=0
        w_cbf = 1.0 - w_cf  # rinormalizzazione: CBF assorbe tutto il peso residuo
        return torch.stack([w_cf, w_cbf], dim=-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
