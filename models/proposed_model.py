import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
            mask: [batch_size, seq_len] (1=valid, 0=padding)
        Returns:
            [batch_size, dim]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        output = torch.matmul(attn_weights, V)
        output = output.mean(dim=1)

        return output


class IndividualGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = SelfAttention(hidden_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_size]
            lengths: [batch_size] (actual sequence lengths)
        Returns:
            [batch_size, hidden_size]
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        mask = torch.zeros(out.shape[:2], device=out.device)
        for i, l in enumerate(lengths):
            if l > 0:
                mask[i, :l] = 1.0

        attended = self.attention(out, mask)
        return attended


class DialogManagementUnit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, interaction_hidden: torch.Tensor,
                speaker_ids: list, target_speaker: str,
                interlocutor: str) -> dict:
        """Interaction GRUのhidden stateを役割ごとに振り分け

        Args:
            interaction_hidden: [seq_len, hidden_size]
            speaker_ids: list of speaker_id per timestep
            target_speaker: str
            interlocutor: str or None
        Returns:
            dict with 'speaker', 'interlocutor' tensors
        """
        speaker_h = []
        interlocutor_h = []

        for t, sid in enumerate(speaker_ids):
            h = interaction_hidden[t].unsqueeze(0)
            if sid == target_speaker:
                speaker_h.append(h)
            elif sid == interlocutor:
                interlocutor_h.append(h)

        result = {}
        if len(speaker_h) > 0:
            result['speaker'] = torch.cat(speaker_h, dim=0)
        else:
            result['speaker'] = torch.zeros(
                1, interaction_hidden.shape[1],
                device=interaction_hidden.device
            )

        if len(interlocutor_h) > 0:
            result['interlocutor'] = torch.cat(interlocutor_h, dim=0)
        else:
            result['interlocutor'] = torch.zeros(
                1, interaction_hidden.shape[1],
                device=interaction_hidden.device
            )

        return result


class ProposedModel(nn.Module):
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

        self.interlocutor_gru = IndividualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.dialog_mgmt = DialogManagementUnit()

        self.fc = nn.Linear(hidden_size * 3, num_classes)

    def forward(self, context_features: torch.Tensor,
                context_lengths: torch.Tensor,
                context_speaker_ids: list,
                roles: list) -> torch.Tensor:
        """
        Args:
            context_features: [batch_size, seq_len, input_size]
            context_lengths: [batch_size]
            context_speaker_ids: list of list of str
            roles: list of dict with 'speaker', 'interlocutor'
        Returns:
            [batch_size, num_classes]
        """
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
        hI_list = []
        hA_list = []

        for b in range(batch_size):
            actual_len = context_lengths[b].item()
            h_interaction = interaction_out[b, :actual_len]
            speaker_ids = context_speaker_ids[b][:actual_len]
            role = roles[b]

            dm_output = self.dialog_mgmt(
                h_interaction, speaker_ids,
                role['speaker'], role.get('interlocutor')
            )

            speaker_h = dm_output['speaker']
            interlocutor_h = dm_output['interlocutor']

            speaker_lengths = torch.tensor([speaker_h.shape[0]])
            interlocutor_lengths = torch.tensor([interlocutor_h.shape[0]])

            hS = self.speaker_gru(
                speaker_h.unsqueeze(0), speaker_lengths
            )
            hI = self.interlocutor_gru(
                interlocutor_h.unsqueeze(0), interlocutor_lengths
            )

            hA = interaction_hidden[-1, b].unsqueeze(0)

            hS_list.append(hS)
            hI_list.append(hI)
            hA_list.append(hA)

        hS = torch.cat(hS_list, dim=0)
        hI = torch.cat(hI_list, dim=0)
        hA = torch.cat(hA_list, dim=0)

        h = torch.cat([hS, hI, hA], dim=-1)
        logits = self.fc(h)

        return logits