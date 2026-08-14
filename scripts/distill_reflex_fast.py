"""
HIGH-PERFORMANCE VECTORIZED KNOWLEDGE DISTILLATION ENGINE
Distills knowledge from Prajna 264.7M Foundation Teacher into 25k Fast-Reflex Student.
"""
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.models.pinn_foundation import PrajnaFoundationPINN
from prajna_core.models.fast_reflex import PrajnaFastReflex

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 3.0, alpha_kd: float = 0.6, alpha_ce: float = 0.4):
        super().__init__()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.alpha_ce = alpha_ce
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss_kd = self.kl_div(soft_student, soft_targets) * (self.temperature ** 2)
        loss_ce = self.ce_loss(student_logits, targets)
        return self.alpha_kd * loss_kd + self.alpha_ce * loss_ce
