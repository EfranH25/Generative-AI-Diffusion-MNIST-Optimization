import torch
import torch.nn as nn
import torch.nn.functional as F 

try:
    import bitsandbytes as bnb
    print('imported bitsandbytes')
    
except:
    print('cant import bitsandbytes')
    bnb = None

class EMA:
    """Implementation of Exponential Moving Average (EMA) to be used with UNet or UNet + CFG
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

#? check around 10:30 in vid 2
class SelfAttention(nn.Module):
    def __init__(self, channels, size, quant=False):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        if quant:
          self.ff_self = nn.Sequential(
              nn.LayerNorm([channels]),
              bnb.nn.Linear8bitLt(channels, channels),
              nn.GELU(),
              bnb.nn.Linear8bitLt(channels, channels),
          )
        else:
          self.ff_self = nn.Sequential(
              nn.LayerNorm([channels]),
              nn.Linear(channels, channels),
              nn.GELU(),
              nn.Linear(channels, channels),
          )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, groups=[1,1]):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, groups=groups[0]),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=groups[0]),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256,quant=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        if quant:
          self.emb_layer = nn.Sequential(
              nn.SiLU(),
              bnb.nn.Linear8bitLt(
                  emb_dim,
                  out_channels
              ),
          )
        else:
          self.emb_layer = nn.Sequential(
              nn.SiLU(),
              nn.Linear(
                  emb_dim,
                  out_channels
              ),
          )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, quant=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        if quant:
          self.emb_layer = nn.Sequential(
              nn.SiLU(),
              bnb.nn.Linear8bitLt(
                  emb_dim,
                  out_channels
              ),
          )
        else:
          self.emb_layer = nn.Sequential(
              nn.SiLU(),
              nn.Linear(
                  emb_dim,
                  out_channels
              ),
          )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, size = 64, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        if device.lower() == 'cuda' and torch.cuda.is_available():
            print("CUDAAAAAAAAAAA")
            self.use_cuda = torch.device('cuda')
        else:
            print('using CPU')
            self.use_cuda = torch.device('cpu')
        
        #? used for UNet. Has an encoder, a bottleneck, and a decoder
        #? check vid 2 at 7:24 for explaination
        
        #? down sampler
        
        size2, size4, size8 = [int(size*2), int(size*4), int(size*8)]
        size_2, size_4, size_8 = [size//2, size//4, size//8]
        #print(size2, size4, size8)
        #print(size_2, size_4, size_8)
        
        self.inc = DoubleConv(c_in, size) # (c_in, 64)
        self.down1 = Down(size, size2, emb_dim=size4) # (64, 128)
        self.sa1 = SelfAttention(size2, size_2) # (128, 32)
        self.down2 = Down(size2, size4, emb_dim=size4) # (128, 256)
        self.sa2 = SelfAttention(size4, size_4) # (256, 16)
        self.down3 = Down(size4, size4, emb_dim=size4) # (256, 256)
        self.sa3 = SelfAttention(size4, size_8) # (256, 8)

        #? bottleneck
        self.bot1 = DoubleConv(size4, size8) # (256, 512)
        self.bot2 = DoubleConv(size8, size8) # (512, 512)
        self.bot3 = DoubleConv(size8, size4) # (512, 256)

        #? decoder --> reverse of down sampler so has 3 upsampling blocks with self attention block in between each up sampler
        self.up1 = Up(size8, size2, emb_dim=size4) # (512, 128)
        self.sa4 = SelfAttention(size2, size_4) # (128, 16)
        self.up2 = Up(size4, size, emb_dim=size4) # (256, 64)
        self.sa5 = SelfAttention(size, size_2) # (64, 32)
        self.up3 = Up(size2, size, emb_dim=size4) # (128, 64)
        self.sa6 = SelfAttention(size, size) # (64, 64)
        
        #? project back out to output channel dimensions
        self.outc = nn.Conv2d(size, c_out, kernel_size=1) # (64, c_out, kernel_size=1)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        """_summary_

        Args:
            x (_type_): noised image
            t (_type_): timestep
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        #? downsample
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        #? bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        #? upsample
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        
        output = self.outc(x)
        return output

class UNet_conditional(nn.Module):
    """UNet used for diffusion. Has Classifier Free Guadience (CFG) Implemented.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, args, c_in=3, c_out=3, size = 64, time_dim=256, num_classes=None, device="cuda", USE_GPU = True, groups=[1, 1, 1, 1, 1, 1, 1, 1, 1]):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        
        if device.lower() == 'cuda' and torch.cuda.is_available():
            print("CUDAAAAAAAAAAA")
            self.use_cuda = torch.device('cuda')
        else:
            print('using CPU')
            self.use_cuda = torch.device('cpu')
        
        if 'quant' in args.optim:
            self.quant = True
            print('using quanitzation optimization Unet')
        else:
          self.quant = False

        if 'prune' in args.optim:
            self.prune = True
            print('using pruning optimization Unet')
        else:
          self.prune = False
        
        if 'deep' in args.optim:
            self.groups = groups
            print('using deepwise convolution optimization Unet')
        else:
            print('using default groups')
            self.groups = [1, 1, 1, 1, 1, 1, 1, 1, 1]
            
        print('using these groups:', self.groups)
        size2, size4, size8 = [int(size*2), int(size*4), int(size*8)]
        size_2, size_4, size_8 = [size//2, size//4, size//8]
        #print(size2, size4, size8)
        #print(size_2, size_4, size_8)
        
        #? Sets up neural network architecture
        self.inc = DoubleConv(c_in, size, groups=[self.groups[0], self.groups[1]])
        self.down1 = Down(size, size2, emb_dim=size4, quant=self.quant)
        self.sa1 = SelfAttention(size2, size_2, quant=self.quant)
        self.down2 = Down(size2, size4, emb_dim=size4, quant=self.quant)
        self.sa2 = SelfAttention(size4, size_4, quant=self.quant)
        self.down3 = Down(size4, size4, emb_dim=size4, quant=self.quant)
        self.sa3 = SelfAttention(size4, size_8, quant=self.quant)

        self.bot1 = DoubleConv(size4, size8, groups=[self.groups[2], self.groups[3]])
        self.bot2 = DoubleConv(size8, size8, groups=[self.groups[4], self.groups[5]])
        self.bot3 = DoubleConv(size8, size4, groups=[self.groups[6], self.groups[7]])

        self.up1 = Up(size8, size2, emb_dim=size4, quant=self.quant)
        self.sa4 = SelfAttention(size2, size_4, quant=self.quant)
        self.up2 = Up(size4, size, emb_dim=size4, quant=self.quant)
        self.sa5 = SelfAttention(size, size_2, quant=self.quant)
        self.up3 = Up(size2, size, emb_dim=size4, quant=self.quant)
        self.sa6 = SelfAttention(size, size, quant=self.quant)
        self.outc = nn.Conv2d(size, c_out, kernel_size=1, groups=self.groups[8])

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output