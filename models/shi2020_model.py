import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.proposed_model import SelfAttention, IndividualGRU


class Shi2020Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

        self.interaction_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.speaker_gru = IndividualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.other_gru = IndividualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size * 3, num_classes)

    def forward(self, context_features: torch.Tensor,
                context_lengths: torch.Tensor,
                context_speaker_ids: list,
                roles: list, **kwargs) -> torch.Tensor:
        batch_size = context_features.shape[0]

        packed = nn.utils.rnn.pack_padded_sequence(
            context_features, context_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        packed_out, interaction_hidden = self.interaction_gru(packed)
        interaction_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        hS_list = []
        hO_list = []
        hA_list = []

        for b in range(batch_size):
            actual_len = context_lengths[b].item()
            h_interaction = interaction_out[b, :actual_len]
            speaker_ids = context_speaker_ids[b][:actual_len]
            target_speaker = roles[b]['speaker']

            speaker_h = []
            other_h = []
            for t, sid in enumerate(speaker_ids):
                h = h_interaction[t].unsqueeze(0)
                if sid == target_speaker:
                    speaker_h.append(h)
                else:
                    other_h.append(h)

            if len(speaker_h) == 0:
                speaker_h = [torch.zeros(
                    1, self.hidden_size, device=context_features.device)]
            if len(other_h) == 0:
                other_h = [torch.zeros(
                    1, self.hidden_size, device=context_features.device)]

            speaker_tensor = torch.cat(speaker_h, dim=0)
            other_tensor = torch.cat(other_h, dim=0)

            s_len = torch.tensor([speaker_tensor.shape[0]])
            o_len = torch.tensor([other_tensor.shape[0]])

            hS = self.speaker_gru(speaker_tensor.unsqueeze(0), s_len)
            hO = self.other_gru(other_tensor.unsqueeze(0), o_len)
            hA = interaction_hidden[-1, b].unsqueeze(0)

            hS_list.append(hS)
            hO_list.append(hO)
            hA_list.append(hA)

        hS = torch.cat(hS_list, dim=0)
        hO = torch.cat(hO_list, dim=0)
        hA = torch.cat(hA_list, dim=0)

        h = torch.cat([hS, hO, hA], dim=-1)
        logits = self.fc(h)
        return logits