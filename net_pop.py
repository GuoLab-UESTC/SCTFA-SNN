from parameters import *

class snn(nn.Module):
    def __init__(self, sctfa):
        super(snn, self).__init__()

        self.sctfa = sctfa

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_planes)
        if self.sctfa:
            self.att1 = attention(out_planes, out_planes)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_planes)
        if self.sctfa:
            self.att2 = attention(out_planes, out_planes)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_planes)
        if self.sctfa:
            self.att3 = attention(out_planes, out_planes)

        self.fc1 = nn.Linear(cfg_kernel[-1][0] * cfg_kernel[-1][1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1] * 5)

    def forward(self, inputs, time_window=20):
        batch_size = inputs.size(0)
        c1_spike = c1_mem = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0][0], cfg_kernel[0][1], device=device)
        c2_spike = c2_mem = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1][0], cfg_kernel[1][1], device=device)
        c3_spike = c3_mem = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2][0], cfg_kernel[2][1], device=device)

        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1] * 5, device=device)

        for step in range(time_window):
            x = inputs[:, step].float()
            
            if self.sctfa:
                c1_mem = torch.mul(c1_mem, self.att1.mixed(c1_spike))
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike, BN=self.bn1)
            out = F.avg_pool2d(c1_spike, 2)

            if self.sctfa:
                c2_mem = torch.mul(c2_mem, self.att2.mixed(c2_spike))
            c2_mem, c2_spike = mem_update(self.conv2, out, c2_mem, c2_spike, BN=self.bn2)
            out = F.avg_pool2d(c2_spike, 2)

            if self.sctfa:
                c3_mem = torch.mul(c3_mem, self.att3.mixed(c3_spike))
            c3_mem, c3_spike = mem_update(self.conv3, out, c3_mem, c3_spike, BN=self.bn3)
            out = F.avg_pool2d(c3_spike, 2)


            out = out.view(batch_size, -1)
            h1_mem, h1_spike = mem_update(self.fc1, out,      h1_mem, h1_spike)
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)

            h2_sumspike += h2_spike

        outputs = F.avg_pool1d(h2_sumspike.unsqueeze(1), 5).squeeze(1)/time_window
        return outputs

