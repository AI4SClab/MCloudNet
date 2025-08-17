from torch import nn
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class MaskModel(nn.Module):
    def __init__(self, input_shape=(32, 25, 96), wavelet_basic='db4', decomp_level=4, mspi_len=14, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.mask = nn.Parameter(torch.empty((input_shape[0], input_shape[1], input_shape[2]), device=device))
        torch.nn.init.xavier_normal_(self.mask)

        self.dwt_1d = DWT1DForward(wave=wavelet_basic, J=decomp_level)
        self.idwt_1d = DWT1DInverse(wave=wavelet_basic)
        self.threshold_tensors = nn.ParameterList([nn.Parameter(torch.full((input_shape[0], input_shape[1], 1), 0.1, device=device)) for _ in range(decomp_level)])

        self.scale_factor = 10

    def forward(self, x):
        B, C, L = x.shape
        # 1D DWT
        xl, xh = self.dwt_1d(x)
        for i in range(len(xh)):
            xh[i] = torch.sign(xh[i]) * torch.max(abs(xh[i]) - self.threshold_tensors[i], torch.tensor(0.0, device=x.device))
        z = self.idwt_1d((xl, xh))
        if z.shape[2] > L:
            z = z[:, :, :L]      
            
        mask = torch.sigmoid(self.mask * self.scale_factor)
        z = z * mask
        
        return z

    
