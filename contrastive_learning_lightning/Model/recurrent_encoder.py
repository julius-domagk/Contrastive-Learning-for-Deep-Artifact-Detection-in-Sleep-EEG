from torch import nn
import torch

class RecurrentEncoder(nn.Module):
    def __init__(self, first_hidden_dim=256, gru_residual_dim=128, final_dim=4):
        super().__init__()
        
        hidden_dims=(first_hidden_dim, first_hidden_dim//2, first_hidden_dim//4)

        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.gru_full = nn.GRU(input_size=1, hidden_size=hidden_dims[0], batch_first=True)
        self.gru_half = nn.GRU(input_size=1, hidden_size=hidden_dims[1], batch_first=True)
        self.gru_quarter = nn.GRU(input_size=1, hidden_size=hidden_dims[2], batch_first=True)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
        self.fc1 = nn.Linear(sum(hidden_dims), gru_residual_dim)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(gru_residual_dim)
        
        self.gru_residual = nn.GRU(input_size=gru_residual_dim, hidden_size=gru_residual_dim, batch_first=True)        
        self.fc_out = nn.Linear(gru_residual_dim, final_dim)

    def forward(self, x):

        B, C, L = x.size()
        assert C == 1, "Input must have one channel."
        
        x_seq = x.permute(0, 2, 1)

        out1, _ = self.gru_full(x_seq)

        x_half = self.downsample(x).permute(0, 2, 1)
        out2, _ = self.gru_half(x_half)
        out2 = self.upsample(out2.permute(0, 2, 1)).permute(0, 2, 1)

        x_quarter = self.downsample(self.downsample(x)).permute(0, 2, 1)
        out3, _ = self.gru_quarter(x_quarter)
        out3 = self.upsample(out3.permute(0, 2, 1)).permute(0, 2, 1)
        out3 = self.upsample(out3.permute(0, 2, 1)).permute(0, 2, 1)

        min_len = min(out1.shape[1], out2.shape[1], out3.shape[1])
        out1, out2, out3 = out1[:, :min_len], out2[:, :min_len], out3[:, :min_len]
        concat = torch.cat([out1, out2, out3], dim=-1)

        x_proj = self.relu(self.fc1(concat))
        residual = x_proj
        x_norm = self.ln(x_proj)
        gru_out, _ = self.gru_residual(x_norm)
        x = gru_out + residual
        out = self.fc_out(x)
        
        return out
