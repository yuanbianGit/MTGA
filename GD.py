import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from util.spectral import SpectralNorm
from util.gumbel import gumbel_softmax
import numpy as np
import math
from torch.distributions import Normal, Uniform
import random

class Pat_Discriminator(nn.Module):
  """
  Defines a PatchGAN discriminator
  Code based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
  """
  def __init__(self, input_nc, ndf=64, n_layers=3, norm='bn'):
    """Construct a PatchGAN discriminator
    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """
    super(Pat_Discriminator, self).__init__()

    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
      use_bias = norm_layer.func != nn.BatchNorm2d
    else:
      use_bias = norm_layer != nn.BatchNorm2d

    kw = 4
    padw = 1
    sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):  # gradually increase the number of filters
      nf_mult_prev = nf_mult
      nf_mult = min(2 ** n, 8)
      sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
    sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
    self.model = nn.Sequential(*sequence)

  def forward(self, x):
    return self.model(x), torch.ones_like(x)


class MS_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3, norm='bn', num_D=3, temperature=-1, use_gumbel=False):
    super(MS_Discriminator, self).__init__()
    self.num_D = num_D
    self.n_layers = n_layers
    self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, count_include_pad=False)
    self.same0 = SamePadding(kernel_size=3, stride=2)
    self.same1 = SamePadding(kernel_size=4, stride=2)
    self.same2 = SamePadding(kernel_size=4, stride=1)
    self.Mask = Mask(norm, temperature, use_gumbel)
    # 由subDiscriminator创建三个分支的大D
    for i in range(num_D):
      netD = sub_Discriminator(input_nc, ndf, n_layers, norm)
      for j in range(n_layers+2): setattr(self, 'D'+str(i)+'_layer'+str(j), getattr(netD, 'layer'+str(j)))                                   

  def single_forward(self, model, x):
    result = [x]
    for i in range(len(model)): 
      samepadding = self.same1 if i < len(model)-2 else self.same2
      result.append(model[i](samepadding(result[-1])))
    return result[1:]

  def forward(self, x):        
    num_D = self.num_D
    proposal = []  # 放模型每个阶段输出的特征
    result = []  #放sub-model最后一层输出特征
    mask = None
    input_downsampled = x
    for i in range(num_D):
      model = [getattr(self, 'D'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
      proposal.append(self.single_forward(model, input_downsampled)) #[[D2L0, D2L1,..., D2L4],[D1L0,...,D1L4],[D0L0,...,D0L4]]
      if i != (num_D-1): input_downsampled = self.downsample(self.same0(input_downsampled))
    for i in proposal: result.append(i[-1])
    mask = self.Mask(x, proposal)
    return result, mask
        
# (64,128,256,512,1) 
class sub_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3, norm='in'):
    super(sub_Discriminator, self).__init__()
    self.n_layers = n_layers

    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    sequence = [[SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, bias=use_bias)), nn.LeakyReLU(0.2, True)]]
    nf = ndf
    for n in range(1, n_layers):
      nf_prev = nf
      nf = min(nf*2, 512)
      sequence += [[SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, bias=use_bias)), norm_layer(nf), nn.LeakyReLU(0.2, True)]]

    nf_prev = nf
    nf = min(nf*2, 512)
    sequence += [[SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, bias=use_bias)), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
    sequence += [[nn.Conv2d(nf, 1, kernel_size=4, stride=1)]]

    for n in range(len(sequence)):
      setattr(self, 'layer'+str(n), nn.Sequential(*sequence[n]))

  def forward(self, input):
    res = [input]
    for n in range(self.n_layers+2):
      model = getattr(self, 'layer'+str(n))
      res.append(model(res[-1]))
    return res[1:]

class Mask(nn.Module):
  def __init__(self, norm, temperature, use_gumbel, fused=1):
    super(Mask, self).__init__()
    self.temperature = temperature
    self.use_gumbel = use_gumbel
    self.fused = fused
    self.T = nn.Parameter(torch.Tensor([1]))
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    small_channels = [512, 512, 256, 128]
    big_channels = [512+128, 512+128+64, 128+64, 64] if self.fused == 2 else [512, 512, 128, 64]

    self.up32_16 = UpLayer(big_channels=big_channels[0], out_channels=512, small_channels=small_channels[0], norm_layer=norm_layer)
    self.up16_8 = UpLayer(big_channels=big_channels[1], out_channels=256, small_channels=small_channels[1], norm_layer=norm_layer)
    self.up8_4 = UpLayer(big_channels=big_channels[2], out_channels=128, small_channels=small_channels[2], norm_layer=norm_layer)
    # self.up4_2 = UpLayer(big_channels=big_channels[3], out_channels=64, small_channels=small_channels[3], norm_layer=norm_layer)
    self.deconv1 = nn.Sequential(*[SpectralNorm(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)), nn.LeakyReLU(0.2, True)])
    self.deconv2 = nn.Sequential(*[SpectralNorm(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)), nn.LeakyReLU(0.2, True)])
    self.conv2 = nn.Sequential(*[nn.Conv2d(128, 1, kernel_size=1, stride=1)])
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, x, proposal):
    n,c,h,w = x.size()
    if self.temperature == -1: return torch.ones((n,1,h,w)).cuda()
    scale32 = proposal[2][3]
    scale16 = torch.cat((proposal[2][1], proposal[1][3]),1) if self.fused == 2 else proposal[1][3]
    scale8 = torch.cat((proposal[0][3], proposal[1][1], proposal[2][0]),1) if self.fused == 2 else proposal[0][3]
    scale4 = torch.cat((proposal[0][1], proposal[1][0]),1) if self.fused == 2 else proposal[0][1]
    scale2 = proposal[0][0]
    out = self.up32_16(scale32, scale16)
    out = self.up16_8(out, scale8)
    out = self.up8_4(out, scale4)
    # out = self.up4_2(out, scale2)
    out = self.deconv1(out) 
    out = self.deconv2(out) 
    out = self.conv2(out)

    if not self.use_gumbel:
      logits = self.logsoftmax(out.view(n, -1))
      th, _ = torch.topk(logits, k=int(self.temperature), dim=1, largest=True)
      mask, zeros, ones = torch.zeros_like(logits).cuda(), torch.zeros(h*w).cuda(), torch.ones(h*w).cuda()
      for i in range(n):
        mask[i,:] = torch.where(logits[i,:]>=th[i, int(self.temperature)-1], ones, zeros)
      mask = mask.view(n, 1, h, w)
    elif self.use_gumbel:
      logits = gumbel_softmax(out.view(n, -1), k=int(self.temperature), T=self.T, hard=True, eps=1e-10).view(n, 1, h, w)
      mask = logits.cuda()
      # logits = F.gumbel_softmax(out.view(n, -1), tau=self.temperature).view(n, 1, h, w)
      # # logits_normed = torch.clamp((logits_normed+1e-4), min=0, max=1)
      # logits = np.minimum(1.0, logits.data.cpu().numpy()*(h*w)+1e-4)
      # mask = torch.bernoulli(torch.from_numpy(logits)).cuda()
    return mask

class UpLayer(nn.Module):
  def __init__(self, big_channels, out_channels, small_channels, norm_layer):
    super(UpLayer, self).__init__()
    self.big_channels = big_channels
    self.out_channels = out_channels
    self.small_channels = small_channels
    self.conv1 = nn.Sequential(*[SpectralNorm(nn.Conv2d(self.big_channels, self.small_channels, kernel_size=1, stride=1)), norm_layer(self.small_channels), nn.LeakyReLU(0.2, True)])
    self.conv2 = nn.Sequential(*[SpectralNorm(nn.Conv2d(self.small_channels, self.out_channels, kernel_size=3, stride=1, padding=1)), norm_layer(self.out_channels), nn.LeakyReLU(0.2, True)])

  def forward(self, small, big):
    small = F.upsample(small, size=(big.size()[2], big.size()[3]), mode='bilinear')
    big = self.conv1(big)
    out = self.conv2(big+small)
    return out

class MixCamStyle(nn.Module):
  """MixStyle.
  Reference:
    Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
  """

  def __init__(self, p=0.5, alpha=0.9, eps=1e-6, mix='random'):
    """
    Args:
      p (float): probability of using MixStyle.
      alpha (float): parameter of the Beta distribution.
      eps (float): scaling parameter to avoid numerical issues.
      mix (str): how to mix.
    """
    super().__init__()
    self.p = p
    self.beta = torch.distributions.Beta(alpha, alpha)
    self.eps = eps
    self.alpha = alpha
    self.mix = mix
    self._activated = True

  def __repr__(self):
    return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

  def set_activation_status(self, status=True):
    self._activated = status

  def update_mix_method(self, mix='random'):
    self.mix = mix

  def forward(self, x, camids):
    if not self.training or not self._activated:
      return x

    if random.random() > self.p:
      return x

    B = x.size(0)

    mu = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    sig = (var + self.eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    x_normed = (x - mu) / sig

    # 对应duke数据集，8个摄像头
    # torch.where(camids == 1, 1, 0)
    IDS=camids.unsqueeze(dim=0).repeat(8,1)
    cam_matrix=torch.range(0,(8-1)).unsqueeze(dim=1).repeat(1,B).cuda()
    cam_bool = torch.where(IDS == cam_matrix, 1, 0)  #(8,64)
    
    mu_cam = torch.matmul(cam_bool.float(),mu.squeeze()) #(8,32)
    mu_cam_mean = mu_cam /torch.sum(cam_bool+self.eps,dim=1).unsqueeze(dim=1)  #(8,32)
    sig_cam = torch.matmul(cam_bool.float(),sig.squeeze()) #(8,32)
    sig_cam_mean = sig_cam /torch.sum(cam_bool+self.eps,dim=1).unsqueeze(dim=1)  #(8,32)

    #解决如果这批中没有该摄像头的情况，也就是sig和mu_cam=0
    non_cam = torch.where(sig_cam_mean.sum(dim=1)<0.1)[0]
    sig_cam_mean[non_cam] = torch.sum(sig_cam_mean,dim=0)/(sig_cam_mean.size(0)-non_cam.size(0))
    mu_cam_mean[non_cam] = torch.sum(mu_cam_mean,dim=0)/(mu_cam_mean.size(0)-non_cam.size(0))


    ids_shuffle = (camids+torch.randint(1,7,(1,)).cuda())%8
    mu_cam_mean_shuffle = mu_cam_mean[ids_shuffle].unsqueeze(dim=2).unsqueeze(dim=3)  #(64.32.1.1)
    sig_cam_mean_shuffle = sig_cam_mean[ids_shuffle].unsqueeze(dim=2).unsqueeze(dim=3)  #(64.32.1.1)

    lmda = self.beta.sample((B, 1, 1, 1))
    lmda = lmda.to(x.device)
    mu_mix = mu * lmda + mu_cam_mean_shuffle * (1 - lmda)
    sig_mix = sig * lmda + sig_cam_mean_shuffle * (1 - lmda)

    return x_normed * sig_mix + mu_mix


class LockStyle(nn.Module):
  """MixStyle.
  Reference:
    Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
  """

  def __init__(self, p=0.5, alpha=0.9, eps=1e-6, mix='random'):
    """
    Args:
      p (float): probability of using MixStyle.
      alpha (float): parameter of the Beta distribution.
      eps (float): scaling parameter to avoid numerical issues.
      mix (str): how to mix.
    """
    super().__init__()
    self.p = p
    self.beta = torch.distributions.Beta(alpha, alpha)
    self.eps = eps
    self.alpha = alpha
    self.mix = mix
    self._activated = True

  def __repr__(self):
    return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

  def set_activation_status(self, status=True):
    self._activated = status

  def update_mix_method(self, mix='random'):
    self.mix = mix

  def forward(self, x, camids):
    if not self.training or not self._activated:
      return x


    B = x.size(0)

    mu = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    sig = (var + self.eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    x_normed = (x - mu) / sig

    return x_normed
class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.9, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

class Generator(nn.Module):
  def __init__(self, input_nc, output_nc, ngf,  norm='bn', n_blocks=6,beta=0.1):
    super(Generator, self).__init__()
    n_downsampling = n_upsampling = 2
    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
    for i in range(n_upsampling): 
      up_layers.append([])
    # ngf
    '''
    反射填充nn.ReflectionPad2d对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
    '''
    begin_layers = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)), norm_layer(ngf), nn.ReLU(True)]
    # 2ngf, 4ngf
    for i in range(n_downsampling):
      mult = 2**i
      down_layers += [SpectralNorm(nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias)), norm_layer(ngf*mult*2), nn.ReLU(True)]
    # 4ngf
    mult = 2**n_downsampling
    for i in range(n_blocks):
      res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
    # 2ngf, ngf
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers[i] += [SpectralNorm(nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

    end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

    self.l1 = nn.Sequential(*begin_layers)
    self.l2 = nn.Sequential(*down_layers)
    self.l3 = nn.Sequential(*res_layers)
    self.l4_1 = nn.Sequential(*up_layers[0])
    self.l4_2 = nn.Sequential(*up_layers[1])
    self.l5 = nn.Sequential(*end_layers)
    self.style = LockStyle(alpha=beta)
  def forward(self, inputs,stylsTrans=False,inter_layer=None,cam_info=None):
    # inputs out (B，3，256，128)
    if stylsTrans:
      out = self.l1(inputs)
      if inter_layer=='layer_0':
        out = self.style(out,cam_info)
      out = self.l2(out)
      if inter_layer=='layer_1':
        out = self.style(out,cam_info)
      out = self.l3(out)
      if inter_layer=='layer_2':
        out = self.style(out,cam_info)
      out = self.l4_1(out)
      out = self.l4_2(out)
      if inter_layer=='layer_3':
        out = self.style(out,cam_info)
      out = self.l5(out)
      # out = self.style(out)

    else:
      out = self.l1(inputs)
      out = self.l2(out)
      out = self.l3(out)
      out = self.l4_1(out)
      out = self.l4_2(out)
      out = self.l5(out)
    return out

class UBS(nn.Module):

    def __init__(self, p=0.5, rho=1.0, eps=1e-6):
      super().__init__()
      self.p = p
      self.rho = rho
      self.eps = eps

    def __repr__(self):
      return f'UBS(rho={self.rho}, p={self.p})'

    def forward(self, x):
      if not self.training:
        return x

      if random.random() > self.p:
        return x

      B = x.size(0)
      mu = x.mean(dim=[2, 3], keepdim=True)
      var = x.var(dim=[2, 3], keepdim=True)
      sig = (var + self.eps).sqrt()
      mu, sig = mu.detach(), sig.detach()
      x_normed = (x - mu) / sig

      mu_1 = x.mean(dim=[2, 3], keepdim=True)
      std_1 = x.std(dim=[2, 3], keepdim=True)

      mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
      mu_std = mu_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
      std_mu = std_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
      std_std = std_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
      mu_std.data.clamp_(min=self.eps)
      std_std.data.clamp_(min=self.eps)

      Distri_mu = Uniform(mu_mu - self.rho * mu_std, mu_mu + self.rho * mu_std)
      Distri_std = Uniform(std_mu - self.rho * std_std, std_mu + self.rho * std_std)

      mu_b = Distri_mu.sample([B, ])
      sig_b = Distri_std.sample([B, ])
      mu_b = mu_b.unsqueeze(2).unsqueeze(2)
      sig_b = sig_b.unsqueeze(2).unsqueeze(2)
      mu_b, sig_b = mu_b.detach(), sig_b.detach()

      return x_normed * sig_b + mu_b
class Generator_2out(nn.Module):
  def __init__(self, input_nc, ngf, norm='bn', n_blocks=6):
    super(Generator_2out, self).__init__()

    n_downsampling = n_upsampling = 2
    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    begin_layers, down_layers, res_layers, up_layers_rgb,up_layers_gray, end_layers_rgb, end_layers_gray = [], [], [], [], [], [],[]
    for i in range(n_upsampling):
      up_layers_gray.append([])
      up_layers_rgb.append([])
    # ngf
    '''
    反射填充nn.ReflectionPad2d对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
    '''
    begin_layers = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)), norm_layer(ngf), nn.ReLU(True)]
    # 2ngf, 4ngf
    for i in range(n_downsampling):
      mult = 2**i
      down_layers += [SpectralNorm(nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias)), norm_layer(ngf*mult*2), nn.ReLU(True)]
    # 4ngf
    mult = 2**n_downsampling
    for i in range(n_blocks):
      res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
    # 2ngf, ngf
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers_rgb[i] += [SpectralNorm(nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers_gray[i] += [SpectralNorm(nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

    end_layers_rgb += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
    end_layers_gray += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Tanh()]
    self.l1 = nn.Sequential(*begin_layers)
    self.l2 = nn.Sequential(*down_layers)
    self.l3 = nn.Sequential(*res_layers)
    self.l4_rgb_1 = nn.Sequential(*up_layers_rgb[0])
    self.l4_rgb_2 = nn.Sequential(*up_layers_rgb[1])
    self.l4_gray_1 = nn.Sequential(*up_layers_gray[0])
    self.l4_gray_2 = nn.Sequential(*up_layers_gray[1])
    self.l5_rgb = nn.Sequential(*end_layers_rgb)
    self.l5_gray = nn.Sequential(*end_layers_gray)


  def forward(self, inputs):
    # inputs out (B，3，256，128)
    out = self.l1(inputs)
    out = self.l2(out)
    out = self.l3(out)

    out_gray = self.l4_gray_1(out)
    out_gray = self.l4_gray_2(out_gray)
    out_gray = self.l5_gray(out_gray)

    out_rgb = self.l4_rgb_1(out)
    out_rgb = self.l4_rgb_2(out_rgb)
    out_rgb = self.l5_rgb(out_rgb)
    return out_rgb,out_gray


# 和上边的Generator不同的地方就是每层少了SpectralNorm
class ResnetG(nn.Module):
  def __init__(self, input_nc, output_nc, ngf, norm='bn', n_blocks=6):
    super(ResnetG, self).__init__()

    n_downsampling = n_upsampling = 2
    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
    for i in range(n_upsampling): 
      up_layers.append([])
    # ngf
    begin_layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
    # 2ngf, 4ngf
    for i in range(n_downsampling):
      mult = 2**i
      down_layers += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf*mult*2), nn.ReLU(True)]
    # 4ngf
    mult = 2**n_downsampling
    for i in range(n_blocks):
      res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
    # 2ngf, ngf
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers[i] += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

    end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

    self.l1 = nn.Sequential(*begin_layers)
    self.l2 = nn.Sequential(*down_layers)
    self.l3 = nn.Sequential(*res_layers)
    self.l4_1 = nn.Sequential(*up_layers[0])
    self.l4_2 = nn.Sequential(*up_layers[1])
    self.l5 = nn.Sequential(*end_layers)

  def forward(self, inputs):
    out = self.l1(inputs)
    out = self.l2(out)
    out = self.l3(out)
    out = self.l4_1(out)
    out = self.l4_2(out)
    out = self.l5(out)
    return out

# Define a resnet block
class ResnetBlock(nn.Module):
  def __init__(self, dim, norm_layer, use_bias):
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

  def build_conv_block(self, dim, norm_layer, use_bias):
    conv_block = []
    for i in range(2):
      conv_block += [nn.ReflectionPad2d(1)]
      conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)), norm_layer(dim)]
      if i < 1: 
        conv_block += [nn.ReLU(True)]
    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out

# stride控制了输出的size,然后根据kernel_size()将图像padding!
class SamePadding(nn.Module):
  def __init__(self, kernel_size, stride):
    super(SamePadding, self).__init__()
    self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
    self.stride = torch.nn.modules.utils._pair(stride)

  def forward(self, input):
    in_width = input.size()[2]
    in_height = input.size()[3]
    out_width = math.ceil(float(in_width) / float(self.stride[0]))
    out_height = math.ceil(float(in_height) / float(self.stride[1]))
    pad_along_width = ((out_width - 1) * self.stride[0] +
                       self.kernel_size[0] - in_width)
    pad_along_height = ((out_height - 1) * self.stride[1] +
                        self.kernel_size[1] - in_height)
    pad_left = int(pad_along_width / 2)
    pad_top = int(pad_along_height / 2)
    pad_right = pad_along_width - pad_left
    pad_bottom = pad_along_height - pad_top
    return F.pad(input, (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)), 'constant', 0)

  def __repr__(self):
    return self.__class__.__name__

def weights_init(m):
  classname = m.__class__.__name__
  # logger.info(dir(m))
  if classname.find('Conv') != -1:
    if 'weight' in dir(m): 
      m.weight.data.normal_(0.0, 1)
  elif classname.find('BatchNorm2d') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

class GANLoss(nn.Module):
  def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.cuda.FloatTensor):
    super(GANLoss, self).__init__()
    self.real_label = target_real_label
    self.fake_label = target_fake_label
    self.real_label_var = None
    self.fake_label_var = None
    self.Tensor = tensor
    if use_lsgan: self.loss = nn.MSELoss()
    else: self.loss = nn.BCELoss()

  def get_target_tensor(self, input, target_is_real):
    # target_tensor = None
    if target_is_real:
      real_tensor = self.Tensor(input.size()).fill_(self.real_label)
      self.real_label_var = Variable(real_tensor, requires_grad=False)
      target_tensor = self.real_label_var
    else:
      fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
      self.fake_label_var = Variable(fake_tensor, requires_grad=False)
      target_tensor = self.fake_label_var
    return target_tensor

  def __call__(self, input, target_is_real):
    if isinstance(input[0], list):
      loss = 0
      for input_i in input:
        pred = input_i[-1]
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss += self.loss(pred, target_tensor)
      return loss
    else:
      target_tensor = self.get_target_tensor(input[-1], target_is_real)
      # print(target_tensor.size())
      # print(input[-1].size())
      return self.loss(input[-1], target_tensor)