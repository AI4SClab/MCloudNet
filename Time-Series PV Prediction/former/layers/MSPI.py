import torch
import torch.nn as nn

class PyramidPooling1D(nn.Module):
    def __init__(self, mspi_layers, pool_type):
        super(PyramidPooling1D, self).__init__()
        self.mspi_layers = mspi_layers
        self.pool_type = pool_type
        
    def forward(self, y):
        B, C, L = y.size()
        mspi_outputs = []
        # print(self.mspi_layers)
        for i in range(self.mspi_layers):
            i = i + 1
            if self.pool_type == 'max_pool':
                tensor = nn.AdaptiveMaxPool2d(output_size=(C, i * i))(y)
            elif self.pool_type == 'avg_pool':
                tensor = nn.AdaptiveAvgPool2d(output_size=(C, i * i))(y)
            elif self.pool_type == 'min_pool':
                tensor = -(nn.AdaptiveMaxPool2d(output_size=(C, i * i))(-y))
            mspi_outputs.append(tensor)
        y_mspi = torch.cat(mspi_outputs, dim=2)

        return y_mspi


class MSPINet(nn.Module):
    def __init__(self, mspi_layers, pool_type='max_pool'):
        super(MSPINet, self).__init__()
        self.mspi_layers = mspi_layers
        self.pyramidpooling = PyramidPooling1D(mspi_layers, pool_type)

    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)

    def forward(self, y):
        y_mspi = self.pyramidpooling(y)
        y_out = torch.cat([y_mspi, y], 2)
        return y_out




