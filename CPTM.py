import torch
import torch.nn as nn

class CPTM(nn.Module):
    def __init__(self, in_chan, out_chan, ks, dil_rate, nc, tau=8):
        super(CPTM, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(dil_rate)):
            d = dil_rate[i]
            if i == 0:
                conv_layer = nn.Conv1d(in_chan, out_chan, ks, dilation=d, padding=(ks - 1) * d // 2)
            else:
                conv_layer = nn.Conv1d(out_chan, out_chan, ks, dilation=d, padding=(ks - 1) * d // 2)
            self.conv_layers.append(conv_layer)

        self.activation = nn.ReLU()
        self.tau = tau
        self.nc = nc

        self.P1 = nn.Linear(out_chan, nc)
        self.P2 = nn.Linear(nc, nc)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def apply_conv(self, x):
        vals = []
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
            vals.append(x)
        return vals

    def psi_h(self, vals):
        h = torch.mean(torch.stack(vals, dim=1), dim=1)
        h = self.global_avg_pool(h).view(h.size(0), -1)
        h = self.P1(h)
        h = self.activation(h)
        return self.P2(h)

    def comp_agg(self, vals):
        K = len(vals)
        psi_h_out = self.psi_h(vals)

        exp_val = torch.exp(self.tau * psi_h_out)

        softmax_denom = torch.sum(exp_val, dim=1, keepdim=True)
        norm_exp_val = exp_val / softmax_denom

        y = torch.zeros_like(vals[0])
        for k in range(K):
            y += norm_exp_val[:, k:k + 1] * vals[k]
        return y

    def forward(self, x):
        aconv = self.apply_conv(x)
        return self.comp_agg(aconv)

# Example usage
if __name__ == "__main__":
    in_chan = 1
    out_chan = 16
    kernel_size = 3
    dil_rate = [1, 2, 4]
    num_classes = 3
    seq_len = 48

    model = CPTM(in_chan, out_chan, kernel_size, dil_rate, num_classes)

    input = torch.rand(1, in_chan, seq_len)
    output = model(input)

    print("Output shape:", output.shape)
