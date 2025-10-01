from torch import nn
import torch

class EEGProjector(nn.Module):
    def __init__(self, input_dim=4, lstm_dims=(256, 128, 64), dense_layer = 128, output_dim = 32):
        super().__init__()

        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(input_dim, lstm_dims[0], batch_first=True, bidirectional=True)  
        self.lstm2 = nn.LSTM(input_dim, lstm_dims[1], batch_first=True, bidirectional=True)  
        self.lstm3 = nn.LSTM(input_dim, lstm_dims[2], batch_first=True, bidirectional=True)  

        self.norm1 = nn.LayerNorm(2 * lstm_dims[0])
        self.norm2 = nn.LayerNorm(2 * lstm_dims[1])
        self.norm3 = nn.LayerNorm(2 * lstm_dims[2])

        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * lstm_dims[0] + 2 * 2 * lstm_dims[1] + 2 * 2 * lstm_dims[2], dense_layer),
            nn.ReLU(),
            nn.Linear(dense_layer, output_dim)
        )

    def forward(self, x):
        out1, _ = self.lstm1(x)  
        out1 = self.norm1(out1)

        x_ds1 = self.downsample(x.permute(0, 2, 1)).permute(0, 2, 1) 
        out2, _ = self.lstm2(x_ds1)  

        out2 = self.norm2(out2)
        x_ds2 = self.downsample(x_ds1.permute(0, 2, 1)).permute(0, 2, 1)  
        
        out3, _ = self.lstm3(x_ds2) 
        out3 = self.norm3(out3)

        def first_last_concat(out):
            return torch.cat([out[:, 0], out[:, -1]], dim=1) 

        flo = torch.cat([
            first_last_concat(out1),  
            first_last_concat(out2),  
            first_last_concat(out3)   
        ], dim=1)

        out = self.fc(flo)

        return out
