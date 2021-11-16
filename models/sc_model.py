import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
import torch.nn as nn
from . import networks
from . import losses
from . import vision_transformer as vits
import os
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl, MultivariateNormal

class SCModel(BaseModel):
    """
    This class implements the unpaired image translation model with spatially correlative loss
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        :param parser: original options parser
        :return: the modified parser
        """
        parser.set_defaults(no_dropout=True)

        parser.add_argument('--attn_layers', type=str, default='4, 7, 9', help='compute spatial loss on which layers')
        parser.add_argument('--patch_nums', type=float, default=256, help='select how many patches for shape consistency, -1 use all')
        parser.add_argument('--patch_size', type=int, default=64, help='patch size to calculate the attention')
        parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')
        parser.add_argument('--use_norm', action='store_true', help='normalize the feature map for FLSeSim')
        parser.add_argument('--learned_attn', action='store_true', help='use the learnable attention map')
        parser.add_argument('--augment', action='store_true', help='use data augmentation for contrastive learning')
        parser.add_argument('--T', type=float, default=0.07, help='temperature for similarity')
        parser.add_argument('--lambda_spatial', type=float, default=10.0, help='weight for spatially-correlative loss')
        parser.add_argument('--lambda_spatial_idt', type=float, default=0.0, help='weight for idt spatial loss')
        parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='weight for feature consistency loss')
        parser.add_argument('--lambda_style', type=float, default=0.0, help='weight for style loss')
        parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping')
        parser.add_argument('--lambda_gradient', type=float, default=0.0, help='weight for the gradient penalty')

        return parser

    def __init__(self, opt):
        """
        Initialize the translation losses
        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out
        self.loss_names = ['style', 'G_s', 'per', 'D_real', 'D_fake', 'G_GAN']
        # specify the images you want to save/display
        self.visual_names = ['real_A', 'fake_B' , 'real_B','mask']
        # for test remove mask
        # self.visual_names = ['real_A', 'fake_B' , 'real_B']

        # specify the models you want to save to the disk
        self.model_names = ['G', 'D'] if self.isTrain else ['G']
        # define the networks
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
        #                         opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netG = networks.define_transG(self.gpu_ids)
        ### pretrain module
        self.netPre = losses.VGG16().to(self.device)
        self.dino = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.dino.load_state_dict(state_dict, strict=True)
        self.dino.to(self.device)
        # define the training process
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
            if opt.lambda_identity > 0.0 or opt.lambda_spatial_idt > 0.0:
                # only works when input and output images have the same number of channels
                self.visual_names.append('idt_B')
                if opt.lambda_identity > 0.0:
                    self.loss_names.append('idt_B')
                if opt.lambda_spatial_idt > 0.0:
                    self.loss_names.append('G_s_idt_B')
                assert (opt.input_nc == opt.output_nc)
            if opt.lambda_gradient > 0.0:
                self.loss_names.append('D_Gradient')
            self.fake_B_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images


            # define the loss function
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionStyle = losses.StyleLoss().to(self.device)
            self.criterionFeature = losses.VGGLoss().to(self.device)
            ### opt.learned_attn SpatialCorrelativeLoss(use_conv)
            self.criterionSpatial = losses.SpatialCorrelativeLoss(opt.loss_mode, opt.patch_nums, opt.patch_size, opt.use_norm,
                                    opt.learned_attn, gpu_ids=self.gpu_ids, T=opt.T).to(self.device)
            self.normalization = losses.Normalization(self.device)
            # define the contrastive loss
            if opt.learned_attn:
                self.netF = self.criterionSpatial
                self.model_names.append('F')
                self.loss_names.append('spatial')
            else:
                self.set_requires_grad([self.netPre], False)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The learnable spatially-correlative map is defined in terms of the shape of the intermediate, extracted features
        of a given network (encoder or pretrained VGG16). Because of this, the weights of spatial are initialized at the
        first feedforward pass with some input images
        :return:
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.isTrain:
            self.backward_G()
            self.optimizer_G.zero_grad()
            if self.opt.learned_attn:
                self.optimizer_F = torch.optim.Adam([{'params': list(filter(lambda p:p.requires_grad, self.netPre.parameters())), 'lr': self.opt.lr*0.0},
                                        {'params': list(filter(lambda p:p.requires_grad, self.netF.parameters()))}],
                                         lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
                self.optimizer_F.zero_grad()

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps
        :param input: include the data itself and its metadata information
        :return:
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.opt.isTrain and self.opt.augment:
            self.aug_A = input['A_aug' if AtoB else 'B_aug'].to(self.device)
            self.aug_B = input['B_aug' if AtoB else 'A_aug'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0) and self.opt.isTrain else self.real_A
        #### un_guided
        self.fake,self.unguided_mean, self.unguided_sigma, self.posterior_mean, self.posterior_sigma, self.posterior_sample =self.netG(self.real,self.real_B,True)
        self.unguided_distribution = Independent(Normal(self.unguided_mean, self.unguided_sigma), 1)
        ## add dino attention
        w_featmap = self.real_A.shape[-2] // 8
        h_featmap = self.real_A.shape[-1] // 8
        attentionsrc_src_dino = self.dino.get_last_selfattention(self.real_A)
        nh = attentionsrc_src_dino.shape[1]  # number of head
        attentionsrc_src_dino = attentionsrc_src_dino[0, :, 0, 1:].reshape(nh, -1)
        # interpolate
        attentionsrc_src_dino = attentionsrc_src_dino.reshape(nh, w_featmap, h_featmap)
        mask_ = nn.functional.interpolate(attentionsrc_src_dino.unsqueeze(0), scale_factor=8, mode="nearest")[0]
        ### Regulation
        mask = mask_[0]
        mask = (mask - torch.min(mask)) / (torch.max(mask) - torch.min(mask))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.mask = mask.repeat(1, 3, 1, 1).to(self.device)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0) and self.opt.isTrain:
            self.idt_B = self.fake[self.real_A.size(0):]

    def backward_F(self):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        norm_real_A, norm_real_B, norm_fake_B = self.normalization((self.real_A + 1) * 0.5), self.normalization((self.real_B + 1) * 0.5), self.normalization((self.fake_B.detach() + 1) * 0.5)
        if self.opt.augment:
            norm_aug_A, norm_aug_B = self.normalization((self.aug_A + 1) * 0.5), self.normalization((self.aug_B + 1) * 0.5)
            norm_real_A = torch.cat([norm_real_A, norm_real_A], dim=0)
            norm_fake_B = torch.cat([norm_fake_B, norm_aug_A], dim=0)
            norm_real_B = torch.cat([norm_real_B, norm_aug_B], dim=0)
        self.loss_spatial = self.Spatial_Loss(self.netPre, self.mask,norm_real_A, norm_fake_B, norm_real_B)

        self.loss_spatial.backward()

    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator
        :param netD: the discriminator D
        :param real: real images
        :param fake: images generated by a generator
        :return: discriminator loss
        """
        # real
        real.requires_grad_()
        pred_real = netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True, is_dis=True)
        # fake
        pred_fake = netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_dis=True)
        # combined loss
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        # gradient penalty
        if self.opt.lambda_gradient > 0.0:
            self.loss_D_Gradient, _ = losses.cal_gradient_penalty(netD, real, fake, real.device, lambda_gp=self.opt.lambda_gradient)#
            loss_D += self.loss_D_Gradient
        loss_D.backward()

        return loss_D

    def backward_D(self):
        """Calculate the GAN loss for discriminator"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_B, fake_B.detach())

    def backward_G(self):
        """Calculate the loss for generator G_A"""
        l_style = self.opt.lambda_style
        l_per = self.opt.lambda_perceptual
        l_sptial = self.opt.lambda_spatial
        l_idt = self.opt.lambda_identity
        l_spatial_idt = self.opt.lambda_spatial_idt

        # GAN loss
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)
        # different structural loss
        norm_real_A = self.normalization((self.real_A + 1) * 0.5)
        norm_fake_B = self.normalization((self.fake_B + 1) * 0.5)
        norm_real_B = self.normalization((self.real_B + 1) * 0.5)
        self.loss_style = self.criterionStyle(norm_real_B, norm_fake_B) * l_style if l_style > 0 else 0
        self.loss_per = self.criterionFeature(norm_real_A, norm_fake_B) * l_per if l_per > 0 else 0
        self.loss_G_s = self.Spatial_Loss(self.netPre,self.mask, norm_real_A, norm_fake_B, None) * l_sptial if l_sptial > 0 else 0
        # identity loss
        if l_spatial_idt > 0:
            norm_fake_idt_B = self.normalization((self.idt_B + 1) * 0.5)
            self.loss_G_s_idt_B = self.Spatial_Loss(self.netPre,self.mask, norm_real_B, norm_fake_idt_B, None) * l_spatial_idt
        else:
            self.loss_G_s_idt_B = 0
        self.loss_idt_B = self.criterionIdt(self.real_B, self.idt_B) * l_idt if l_idt > 0 else 0

        self.unguided_loss = 0.1 * MoG_KL_Unit_Gaussian(self.unguided_distribution)

        self.likelihood_posterior_loss = 0.1 * -log_prob_modified(self.unguided_distribution,
                                                                    self.posterior_sample).mean()
        self.global_loss=self.unguided_loss+self.likelihood_posterior_loss
        self.loss_G = self.loss_G_GAN + self.loss_style + self.loss_per + self.loss_G_s + self.loss_G_s_idt_B + self.loss_idt_B+ 1*self.global_loss

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # forward
        self.forward()
        if self.opt.learned_attn:
            self.set_requires_grad([self.netF, self.netPre], True)
            self.optimizer_F.zero_grad()
            self.backward_F()
            self.optimizer_F.step()
        # D_A
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # G_A
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        if self.opt.learned_attn:
            self.set_requires_grad([self.netF, self.netPre], False)
        self.backward_G()
        self.optimizer_G.step()

    def Spatial_Loss(self, net, mask,src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        if mask is not None:
            src=src*mask
            tgt=tgt*mask
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):

            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers


def MoG(gaussians):
    means = gaussians.mean
    variances = torch.sum(gaussians.variance, dim=0)  # .to(device)
    size = means.shape[0]
    b = (torch.sum(means, dim=0) / size)  # .to(device)
    b_temp = torch.mm(b.unsqueeze(1), b.unsqueeze(0))
    B = ((torch.diag(variances) / size + torch.mm(means.transpose(0, 1), means) / size - b_temp))  # .unsqueeze(0)
    return MultivariateNormal(b, covariance_matrix=B)


def MoG_KL_Unit_Gaussian(distribution):
    collapsed_multivariate = MoG(distribution)  # Mixture of Gaussian modeling

    unit_cov = torch.eye(collapsed_multivariate.mean.shape[-1]).cuda()  # .unsqueeze(0)
    # print('cov matrix shape:', unit_cov.shape)
    unit_Gaussian = MultivariateNormal(torch.zeros_like(collapsed_multivariate.mean), unit_cov)

    loss = kl.kl_divergence(unit_Gaussian, collapsed_multivariate)

    return loss

def log_prob_modified(distribution, sample):
    var = distribution.stddev ** 2

    log_prob = -((sample - distribution.mean) ** 2) / (2 * var)
    return log_prob