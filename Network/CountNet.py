




import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import numpy as np
import math
import os

try:
    from class_func import Count2Class
except:
    from Network.class_func import Count2Class


# ============================================================================
# 1. base module functions
# ============================================================================
def Gauss_initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ============================================================================
# 2. modules
# ============================================================================
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1),
             torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1,
                 bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(
            inp_dim, out_dim, kernel_size, stride,
            padding=(kernel_size - 1) // 2, bias=bias
        )
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Upsample(nn.Module):
    ""
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cat_out_ch, cat_out_ch, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        low = F.interpolate(low, size=high.shape[-2:], mode='bilinear', align_corners=False)
        low = self.conv1(low)
        x = torch.cat([high, low], dim=1)
        x = self.conv2(x)
        return x


class VGG16_FPN_Backbone(nn.Module):
    ""
    def __init__(self, pretrained=True, VGG16bn=True, Freezebn=False):
        super().__init__()
        self.VGG16bn = VGG16bn
        self.Freezebn = Freezebn

        if self.VGG16bn:
            vgg = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            )
            feats = list(vgg.features.children())


            self.features1 = nn.Sequential(*feats[:24])    # conv3
            self.features2 = nn.Sequential(*feats[24:34])  # conv4
            self.features3 = nn.Sequential(*feats[34:44])  # conv5
        else:
            vgg = models.vgg16(
                weights=models.VGG16_Weights.DEFAULT if pretrained else None
            )
            feats = list(vgg.features.children())


            self.features1 = nn.Sequential(*feats[:17])    # conv3
            self.features2 = nn.Sequential(*feats[17:24])  # conv4
            self.features3 = nn.Sequential(*feats[24:31])  # conv5

        self.upsample1 = Upsample(512, 256, 256 + 512, 512)  # x3 + x2 -> H/16
        self.upsample2 = Upsample(512, 256, 256 + 256, 512)  # feat16 + x1 -> H/8

        Gauss_initialize_weights(self.upsample1)
        Gauss_initialize_weights(self.upsample2)

        if self.Freezebn:
            self.freeze_bn()

    def forward(self, x):
        x1 = self.features1(x)   # conv3, H/8
        x2 = self.features2(x1)  # conv4, H/16
        x3 = self.features3(x2)  # conv5, H/32

        feat = self.upsample1(x3, x2)   # -> H/16
        feat = self.upsample2(feat, x1) # -> H/8
        return feat, x1, x2, x3

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
# class VGG19_FPN_Backbone(nn.Module):
#     """

#     - IF_VGG19bn=True  -> vgg19_bn
#     - IF_VGG19bn=False -> vgg19



#         x1: conv3, H/8
#         x2: conv4, H/16
#         x3: conv5, H/32
#     """
#     def __init__(self, pretrained=True, VGG19bn=True, Freezebn=False):
#         super().__init__()
#         self.VGG19bn = VGG19bn
#         self.Freezebn = Freezebn
#
#         if self.VGG19bn:
#             vgg = models.vgg19_bn(
#                 weights=models.VGG19_BN_Weights.DEFAULT if pretrained else None
#             )
#             feats = list(vgg.features.children())
#
#             # VGG19_BN:
#             # conv1: 0-6
#             # conv2: 7-13
#             # conv3: 14-26
#             # conv4: 27-39
#             # conv5: 40-52
#             self.features1 = nn.Sequential(*feats[:27])    # conv3, H/8
#             self.features2 = nn.Sequential(*feats[27:40])  # conv4, H/16
#             self.features3 = nn.Sequential(*feats[40:53])  # conv5, H/32
#         else:
#             vgg = models.vgg19(
#                 weights=models.VGG19_Weights.DEFAULT if pretrained else None
#             )
#             feats = list(vgg.features.children())
#
#             # VGG19:
#             # conv1: 0-4
#             # conv2: 5-9
#             # conv3: 10-18
#             # conv4: 19-27
#             # conv5: 28-36
#             self.features1 = nn.Sequential(*feats[:19])    # conv3, H/8
#             self.features2 = nn.Sequential(*feats[19:28])  # conv4, H/16
#             self.features3 = nn.Sequential(*feats[28:37])  # conv5, H/32
#
#         self.upsample1 = Upsample(512, 256, 256 + 512, 512)  # x3 + x2 -> H/16
#         self.upsample2 = Upsample(512, 256, 256 + 256, 512)  # feat16 + x1 -> H/8
#
#         Gauss_initialize_weights(self.upsample1)
#         Gauss_initialize_weights(self.upsample2)
#
#         if self.Freezebn:
#             self.freeze_bn()
#
#     def forward(self, x):
#         x1 = self.features1(x)   # conv3, H/8
#         x2 = self.features2(x1)  # conv4, H/16
#         x3 = self.features3(x2)  # conv5, H/32
#
#         feat = self.upsample1(x3, x2)   # -> H/16
#         feat = self.upsample2(feat, x1) # -> H/8
#         return feat, x1, x2, x3
#
#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()


class Residual(nn.Module):
    ""
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)

        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)

        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)

        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        self.need_skip = (inp_dim != out_dim)

    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        return out


# def self_attention(query, key, value):
#     """


#     """
#     attention_scores = torch.matmul(query, key.transpose(-2, -1))
#     attention_weights = F.softmax(attention_scores, dim=-1)
#     attended = torch.matmul(attention_weights, value)
#     return attended
class SelfAttention(nn.Module):
    def __init__(self, in_channels, attn_down=2):
        super().__init__()
        inter_channels = max(1, in_channels // 8)
        self.attn_down = attn_down

        self.query_conv = nn.Conv2d(in_channels, inter_channels, 1)
        self.key_conv   = nn.Conv2d(in_channels, inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()

        if self.attn_down > 1:
            x_attn = F.avg_pool2d(x, kernel_size=self.attn_down, stride=self.attn_down)
        else:
            x_attn = x

        b, c, h, w = x_attn.size()

        query = self.query_conv(x_attn).view(b, -1, h * w).permute(0, 2, 1)
        key   = self.key_conv(x_attn).view(b, -1, h * w)
        value = self.value_conv(x_attn).view(b, -1, h * w).permute(0, 2, 1)

        attention_scores = torch.matmul(query, key) / math.sqrt(query.size(-1))
        attention_weights = self.softmax(attention_scores)
        out = torch.matmul(attention_weights, value)

        out = out.permute(0, 2, 1).contiguous().view(b, C, h, w)

        if self.attn_down > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return identity + self.gamma * out


# class NewFusion_block(nn.Module):
#     def __init__(self, ch_1, ch_2, ch_out):
#         super(NewFusion_block, self).__init__()
#         self.residual = Residual(ch_1 + ch_2, ch_out)
#         self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
#
#     def forward(self, r, f):
#         r = self_attention(r, r, r)
#         f = self.conv(f)
#         fuse = self.residual(torch.cat([r, f], 1))
#         return fuse
class NewFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_out):
        super(NewFusion_block, self).__init__()

        self.self_attn = SelfAttention(ch_1)
        self.residual = Residual(ch_1 + ch_2, ch_out)
        self.conv = nn.Conv2d(in_channels=ch_2, out_channels=ch_2, kernel_size=1)

    def forward(self, r, f):

        r = self.self_attn(r)


        f = self.conv(f)


        fuse = self.residual(torch.cat([r, f], dim=1))
        return fuse


class Segmentation(nn.Module):
    ""
    def __init__(self, num_features_in):
        super(Segmentation, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, 256, 3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(256, 256, 3, padding=1)
        self.act11 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1)
        self.act22 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.act3 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(64, 64, 3, padding=1)
        self.act33 = nn.ReLU(inplace=True)

        self.out = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = x + self.conv11(x)
        x = self.act11(x)

        x = self.act2(self.conv2(x))
        x = x + self.conv22(x)
        x = self.act22(x)

        x = self.act3(self.conv3(x))
        x = x + self.conv33(x)
        x = self.act33(x)

        seg_feature = self.out(x)
        seg_feature = torch.softmax(seg_feature, dim=1)   # (B,2,H/8,W/8)
        return seg_feature


class DepthEncoder(nn.Module):
    def __init__(self, in_ch=1, mid_ch=32, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, d):
        return self.net(d)


class TeacherDepthRouter(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 2, kernel_size=1)

    def forward(self, depth_feat):
        logits = self.head(depth_feat)           # B,2,H,W
        route = torch.softmax(logits, dim=1)     # B,2,H,W
        return route, logits


class RGBProxyRouter(nn.Module):
    def __init__(self, in_ch=512, hidden_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 2, kernel_size=1)
        )

    def forward(self, feat):
        logits = self.net(feat)                  # B,2,H,W
        route = torch.softmax(logits, dim=1)
        return route, logits


class DGSA(nn.Module):
    def __init__(self, in_ch=512, mode='student', dilation=2,
                 alpha_init=0.2, alpha_max=0.5):
        super().__init__()
        self.mode = mode

        # fine branch
        self.fine_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # coarse branch
        self.coarse_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True)
        )

        # teacher depth-guided router
        self.depth_encoder = DepthEncoder(in_ch=1, mid_ch=32, out_ch=64)
        self.teacher_router = TeacherDepthRouter(in_ch=64)

        # student RGB proxy router
        self.student_router = RGBProxyRouter(in_ch=in_ch, hidden_ch=128)

        # fuse two gated branches -> residual
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )


        self.alpha_max = float(alpha_max)
        init_ratio = alpha_init / max(self.alpha_max, 1e-6)
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        init_raw = math.log(init_ratio / (1.0 - init_ratio))
        self.alpha_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_raw) * self.alpha_max

# class DGSA(nn.Module):
#     def __init__(self, in_ch=512, mode='student', dilation=2):
#         super().__init__()
#         self.mode = mode
#
#         # fine branch
#         self.fine_branch = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         # coarse branch
#         self.coarse_branch = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation),
#             nn.ReLU(inplace=True)
#         )
#
#         # teacher depth-guided router
#         self.depth_encoder = DepthEncoder(in_ch=1, mid_ch=32, out_ch=64)
#         self.teacher_router = TeacherDepthRouter(in_ch=64)
#
#         # student RGB proxy router
#         self.student_router = RGBProxyRouter(in_ch=in_ch, hidden_ch=128)
#
#         # fuse two gated branches -> residual
#         self.fuse = nn.Sequential(
#             nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
#             nn.ReLU(inplace=True)
#         )


    def forward(self, feat, depth=None):

        fine = self.fine_branch(feat)  # fine_raw
        coarse = self.coarse_branch(feat)  # coarse_raw

        # 2) route
        if self.mode == 'teacher':
            if depth is None:
                raise ValueError("DGSA teacher mode requires depth input.")

            depth = F.interpolate(
                depth,
                size=feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )


            B = depth.size(0)
            d = depth.view(B, -1)
            dmin = d.min(dim=1)[0].view(B, 1, 1, 1)
            dmax = d.max(dim=1)[0].view(B, 1, 1, 1)
            depth01 = (depth - dmin) / (dmax - dmin + 1e-8)


            a_fine = depth01
            a_coarse = 1.0 - a_fine

            route_prob = torch.cat([a_fine, a_coarse], dim=1)
            route_prob = route_prob.clamp(1e-6, 1.0 - 1e-6)
            route_prob = route_prob / route_prob.sum(dim=1, keepdim=True)
            route_logits = None

        else:
            route_prob, route_logits = self.student_router(feat)

        a_fine = route_prob[:, 0:1, :, :]
        a_coarse = route_prob[:, 1:2, :, :]

        # 3) gating
        fine_gated = fine * a_fine
        coarse_gated = coarse * a_coarse

        dgsa_res = self.fuse(torch.cat([fine_gated, coarse_gated], dim=1))

        alpha = self.get_alpha()
        dgsa_feat = feat + alpha * dgsa_res


        aux = {
            'route_prob': route_prob,
            'route_logits': route_logits,

            'a_fine': a_fine,
            'a_coarse': a_coarse,

            'fine_raw': fine,
            'coarse_raw': coarse,

            'fine_gated': fine_gated,
            'coarse_gated': coarse_gated,

            'dgsa_res': dgsa_res,
            'dgsa_feat': dgsa_feat,
            'alpha': alpha.detach()
        }
        return dgsa_feat, aux


def couple_map(input_score, input_map):
    input_score_1 = input_score[:, 1, :, :]
    dyn_input_shape = input_map.shape
    input_score_1 = torch.unsqueeze(input_score_1, dim=1)
    input_score_1 = input_score_1.repeat([1, dyn_input_shape[1], 1, 1])
    input_attention_score = torch.mul(input_score_1, input_map)
    return input_attention_score


# ============================================================================
# 3. Final CountNet
# ============================================================================
class CountNet(nn.Module):
    def __init__(self, EmbedOpt, class_indice, class2count_spec=None,
                 dgsa_mode='student', dgsa_dilation=2,
                 dgsa_alpha_init=0.2, dgsa_alpha_max=0.5):
        super(CountNet, self).__init__()
        self.class_indice = class_indice
        self.cnum = len(self.class_indice)
        self.class2count = np.zeros((self.cnum))
        self.class2count[1:] = (class_indice[:-1] + class_indice[1:]) / 2
        self.class2count_spec = class2count_spec

        # class center / interval length
        self.ccen = np.zeros(self.cnum)
        self.ccen[0] = self.class_indice[0] / 2
        self.ccen[1:] = (self.class_indice[:-1] + self.class_indice[1:]) / 2

        self.clen = np.zeros(self.cnum)
        self.clen[0] = self.class_indice[0] / 2
        self.clen[1:] = (self.class_indice[1:] - self.class_indice[:-1]) / 2

        self.ccen = torch.from_numpy(self.ccen).float()
        self.clen = torch.from_numpy(self.clen).float()

        # ------------------------------------------------------------------
        # Feature Embedding Network

        # ------------------------------------------------------------------
        self.EmbedOpt = EmbedOpt
        self.backbone = VGG16_FPN_Backbone(
            pretrained=EmbedOpt['load_weights'],
            VGG16bn=EmbedOpt['IF_VGG16bn'],
            Freezebn=EmbedOpt['IF_freeze_bn']
        )

        # regression Network

        self.CmapReg = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1, 1),
            nn.ReLU()
        )
        Gauss_initialize_weights(self.CmapReg)

        self.dgsa = DGSA(
            in_ch=512,
            mode=dgsa_mode,
            dilation=dgsa_dilation,
            alpha_init=dgsa_alpha_init,
            alpha_max=dgsa_alpha_max
        )
        Gauss_initialize_weights(self.dgsa)

        self.newfusion = NewFusion_block(ch_1=512, ch_2=512, ch_out=512)
        Gauss_initialize_weights(self.newfusion)


        self.segmentation = Segmentation(num_features_in=512)
        Gauss_initialize_weights(self.segmentation)


        self.debug_forward_count = 0
        self.debug_max_print = 5

    def forward(self, im, depth=None, return_aux=False):
        H, W = im.shape[-2], im.shape[-1]

        # 1) backbone
        feat, x1, x2, x3 = self.backbone(im)


        dgsa_feat, aux = self.dgsa(feat, depth=depth)

        if (not self.training) and (self.debug_forward_count < self.debug_max_print):
            print(
                f"[DBG] alpha={aux['alpha'].item():.6f} | "
                f"feat_abs={feat.abs().mean().item():.6f} | "
                f"dgsa_abs={dgsa_feat.abs().mean().item():.6f} | "
                f"res_abs={aux['dgsa_res'].abs().mean().item():.6f} | "
                f"fine_raw_abs={aux['fine_raw'].abs().mean().item():.6f} | "
                f"coarse_raw_abs={aux['coarse_raw'].abs().mean().item():.6f} | "
                f"fine_gated_abs={aux['fine_gated'].abs().mean().item():.6f} | "
                f"coarse_gated_abs={aux['coarse_gated'].abs().mean().item():.6f} | "
                f"route_min={aux['route_prob'].min().item():.6f} | "
                f"route_max={aux['route_prob'].max().item():.6f}"
            )
            self.debug_forward_count += 1


        seg_prob_feat = self.segmentation(feat)

        # 4) upsample seg to input size
        output_segmap = F.interpolate(
            seg_prob_feat,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )


        fusion_feature = couple_map(seg_prob_feat, feat)


        pred_dprob_mul = self.newfusion(dgsa_feat, fusion_feature)

        # 7) count map
        Cmap = self.CmapReg(pred_dprob_mul)

        if return_aux:
            return Cmap, output_segmap, aux
        return Cmap, output_segmap

    def test(self, test_im, test_gtden=None, depth=None):
        feat, _, _, _ = self.backbone(test_im)
        dgsa_feat, _ = self.dgsa(feat, depth=depth)
        # seg_prob_feat = self.segmentation(dgsa_feat)
        # fusion_feature = couple_map(seg_prob_feat, dgsa_feat)
        # pred_dprob_mul = self.newfusion(dgsa_feat, fusion_feature)
        seg_prob_feat = self.segmentation(feat)
        fusion_feature = couple_map(seg_prob_feat, feat)
        pred_dprob_mul = self.newfusion(dgsa_feat, fusion_feature)
        test_cmap = self.CmapReg(pred_dprob_mul)
        return test_cmap

    def encode_count(self, cmap):
        cmap = Count2Class(cmap, self.class_indice[:-1])
        return cmap