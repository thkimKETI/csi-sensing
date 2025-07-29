import torch
import torch.nn as nn

class TinyCSINet(nn.Module):
    def __init__(self, num_esp, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(num_esp, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(8*10*10, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# TCN block 정의
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TinyTCNNet(nn.Module):
    def __init__(self, num_esp, num_classes):
        super().__init__()
        self.tcn1 = TemporalBlock(num_esp, 8, kernel_size=3, dilation=1)
        self.tcn2 = TemporalBlock(8, 8, kernel_size=3, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(8*10*114, num_classes)
    def forward(self, x):
        # x: (batch, esp, 180, 114)
        x = x.permute(0, 3, 1, 2)  # (batch, 114, esp, 180)
        x = x.reshape(-1, x.size(2), x.size(3))  # (batch*114, esp, 180)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.pool(x)  # (batch*114, 8, 10)
        x = x.reshape(-1, 8*10*114)
        return self.fc(x)

# Transformer 기반
class TinyTransformerNet(nn.Module):
    def __init__(self, num_esp, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(180*114, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.25)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        # x: (batch, esp, 180, 114)
        b, esp, t, f = x.shape
        x = x.reshape(b, esp, t*f)  # (batch, esp, 180*114)
        x = self.input_proj(x)      # (batch, esp, 128)
        x = x.permute(1, 0, 2)     # (esp, batch, 128) - transformer 입력
        x = self.transformer(x)    # (esp, batch, 128)
        x = x.mean(dim=0)          # (batch, 128)
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (E, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)  # (E, 1, D)

    def forward(self, x):
        # x: (E, B, D)
        E = x.size(0)
        x = x + self.pe[:E].to(x.device)
        return x

class LargeTransformerNet(nn.Module):
    def __init__(self, num_esp, num_classes, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(180 * 114, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=num_esp)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, E, T, F)
        b, e, t, f = x.shape
        x = x.view(b, e, t * f)            # (B, E, 180*114)
        x = self.input_proj(x)             # (B, E, D)
        x = x.permute(1, 0, 2)             # (E, B, D)
        x = self.pos_encoder(x)            # Positional encoding
        x = self.transformer(x)            # (E, B, D)
        x = self.norm(x)                   # (E, B, D)
        x = x.mean(dim=0)                  # (B, D)
        x = self.dropout(x)
        return self.fc(x)                  # (B, num_classes)