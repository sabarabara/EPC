import torch
import torch.nn as nn


class BLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, context_features: torch.Tensor,
                context_lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            context_features, context_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        forward_h = hidden[-2]
        backward_h = hidden[-1]
        last_hidden = torch.cat([forward_h, backward_h], dim=-1)
        logits = self.fc(last_hidden)
        return logits