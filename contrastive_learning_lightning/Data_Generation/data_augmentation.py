import torch
from torch.utils.data import Dataset


def robust_normalize_1d_torch(x):
    x = x.to(torch.float32)
    quantile_1 = torch.quantile(x, 0.01)
    quantile_99 = torch.quantile(x, 0.99)
    x = torch.clamp(x, quantile_1, quantile_99)

    x_median = x.median()
    quantile_25 = torch.quantile(x, 0.25)
    quantile_75 = torch.quantile(x, 0.75)
    iqr = quantile_75 - quantile_25
    scale = torch.where(iqr > 0, iqr, torch.tensor(1.0, dtype=x.dtype, device=x.device))
    return (x - x_median) / scale

def robust_normalize_sigmoid(x):
    x_median = x.median()
    quantile_25 = torch.quantile(x, 0.25)
    quantile_75 = torch.quantile(x, 0.75)
    iqr = quantile_75 - quantile_25
    scale = torch.where(iqr > 0, iqr, torch.tensor(1.0, dtype=x.dtype, device=x.device))
    return torch.sigmoid((x - x_median) / scale)


class ConsecutiveSegmentPairs(Dataset):

    def __init__(self, data, adm, sampling_rate, participants, supervised=False, normalize_with_sigmoid=True):
        super().__init__()

        self.segment_length = 30 * sampling_rate
        self.data = data

        self.normalize = robust_normalize_sigmoid if normalize_with_sigmoid else robust_normalize_1d_torch

        self.segments = []

        for p in participants:

            d = self.data[p-1]          
            C, L = d.shape
            S = L // self.segment_length 

            for c in range(C):
                for s in range(S - 1):  
                    if (not supervised) or (adm[p-1][c, s] == adm[p-1][c, s+1]):
                        self.segments.append((p-1, c, s))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, k):
        p, c, s = self.segments[k]
        data = self.data[p] 

        start_seg1 = s * self.segment_length
        end_seg1 = start_seg1 + self.segment_length

        start_seg2 = end_seg1
        end_seg2 = start_seg2 + self.segment_length

        seg_i_np = data[c, start_seg1:end_seg1]
        seg_j_np = data[c, start_seg2:end_seg2]

        seg_i = torch.as_tensor(seg_i_np, dtype=torch.float32)
        seg_j = torch.as_tensor(seg_j_np, dtype=torch.float32)
        
        x_i = self.normalize(seg_i)
        x_j = self.normalize(seg_j)
        f_i = torch.abs(torch.fft.fft(x_i))
        f_j = torch.abs(torch.fft.fft(x_j))

        return x_i, x_j, f_i, f_j


class SingleSegmentEmbeddings(Dataset):
    def __init__(self, data, adm, sampling_rate, participants, normalize_with_sigmoid=True):
        super().__init__()

        self.segment_length = 30 * sampling_rate
        self.data = data
    
        self.normalize = robust_normalize_sigmoid if normalize_with_sigmoid else robust_normalize_1d_torch

        self.segments = []
        self.adm = adm

        for p in participants:

            d = self.data[p-1]          
            C, L = d.shape
            S = L // self.segment_length 

            for c in range(C):
                for s in range(S):  
                    self.segments.append((p-1, c, s))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, k):
        p, c, s = self.segments[k]
        data = self.data[p] 

        start_seg = s * self.segment_length
        end_seg = start_seg + self.segment_length

        seg_np = data[c, start_seg:end_seg]
        seg = torch.as_tensor(seg_np, dtype=torch.float32)


        x_i = self.normalize(seg)
        f_i = torch.abs(torch.fft.fft(x_i))
        y_i = self.adm[p][c, s]

        return x_i, f_i, y_i