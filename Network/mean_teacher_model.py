# -*- coding: utf-8 -*-
""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


from Network.CountNet import CountNet
from Network.class_func import get_local_count, loss_seg
import os
import sys




class DepthAnythingV2Wrapper(nn.Module):
    ""
    def __init__(self,
                 repo_root: str,
                 weights_path: str,
                 encoder: str = "vitl",
                 device: torch.device = torch.device("cuda"),
                 depth_input_norm: str = "imagenet",
                 normalize_per_image: bool = True):
        super().__init__()
        self.repo_root = repo_root
        self.weights_path = weights_path
        self.encoder = encoder
        self.device = device
        self.depth_input_norm = depth_input_norm
        self.normalize_per_image = normalize_per_image


        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)


        # repo_root/depth_anything_v2/dpt.py
        # repo_root/depth_anything_v2/depth_anything_v2/dpt.py
        DepthAnythingV2 = None
        errs = []
        for mod in ["depth_anything_v2.dpt", "depth_anything_v2.depth_anything_v2.dpt"]:
            try:
                DepthAnythingV2 = __import__(mod, fromlist=["DepthAnythingV2"]).DepthAnythingV2
                break
            except Exception as e:
                errs.append((mod, str(e)))
        if DepthAnythingV2 is None:
            msg = "[DepthAnythingV2Wrapper] import failed. Tried:\n"
            msg += "\n".join([f" - {m}: {e}" for m, e in errs])
            msg += f"\nrepo_root={self.repo_root}"
            raise ImportError(msg)

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        if self.encoder not in model_configs:
            raise ValueError(f"Unknown DepthAnythingV2 encoder: {self.encoder}")

        self.net = DepthAnythingV2(**model_configs[self.encoder])

        ckpt = torch.load(self.weights_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

        missing, unexpected = self.net.load_state_dict(ckpt, strict=False)
        print(f"[DepthAnythingV2] loaded from: {self.weights_path}")
        print(f"[DepthAnythingV2] missing={len(missing)}, unexpected={len(unexpected)}")

        self.net.to(self.device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        x = rgb01
        if self.depth_input_norm == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std

        depth = self.net(x)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.size(1) != 1:
            depth = depth[:, :1, :, :]

        if self.normalize_per_image:
            B = depth.size(0)
            d = depth.reshape(B, -1)
            dmin = d.min(dim=1)[0].reshape(B, 1, 1, 1)
            dmax = d.max(dim=1)[0].reshape(B, 1, 1, 1)
            depth = (depth - dmin) / (dmax - dmin + 1e-8)

        return depth



class ConsistencyLoss(nn.Module):
    ""

    def __init__(self, loss_type='mse'):
        super(ConsistencyLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            return F.l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def exp_rampup(current, rampup_length):
    ""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


class SemiSupervisedCountingModel(nn.Module):
    def __init__(self, opt, loss_dict, loss_w):
        super().__init__()

        self.opt = opt
        self.loss_dict = loss_dict
        self.loss_w = loss_w


        self.student = CountNet(
            opt['EmbedOpt'],
            opt['class_indice'],
            class2count_spec=opt.get('class2count_spec', None),
            dgsa_mode='student',
            dgsa_dilation=opt.get('dgsa_dilation', 2),
            dgsa_alpha_init=opt.get('dgsa_alpha_init', 0.2),
            dgsa_alpha_max=opt.get('dgsa_alpha_max', 0.5)
        )

        self.teacher = CountNet(
            opt['EmbedOpt'],
            opt['class_indice'],
            class2count_spec=opt.get('class2count_spec', None),
            dgsa_mode='teacher',
            dgsa_dilation=opt.get('dgsa_dilation', 2),
            dgsa_alpha_init=opt.get('dgsa_alpha_init', 0.2),
            dgsa_alpha_max=opt.get('dgsa_alpha_max', 0.5)
        )


        self.reset_teacher_from_student()

        self.global_step = 0

        # EMA decay
        self.ema_decay = opt.get('ema_decay', 0.99)

    def forward(self, x, use_teacher=False, depth=None, return_aux=False):
        if use_teacher:
            return self.teacher(x, depth=depth, return_aux=return_aux)
        else:
            return self.student(x, depth=depth, return_aux=return_aux)

    def reset_teacher_from_student(self):
        ""
        with torch.no_grad():
            self.teacher.load_state_dict(copy.deepcopy(self.student.state_dict()), strict=True)
            for p in self.teacher.parameters():
                p.requires_grad = False

    def update_teacher_model(self):
        ""
        with torch.no_grad():
            alpha = min(1.0 - 1.0 / (self.global_step + 1.0), self.ema_decay)
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=(1.0 - alpha))

    def encode_count(self, cmap, use_teacher=False):
        ""
        if use_teacher:
            return self.teacher.encode_count(cmap)
        return self.student.encode_count(cmap)


class SemiSupervisedTrainer:
    ""

    def __init__(self, model, optimizer, opt, loss_dict, loss_w, device):
        self.model = model
        self.optimizer = optimizer
        self.opt = opt
        self.loss_dict = loss_dict
        self.loss_w = loss_w
        self.device = device


        self.consistency_loss_fn = ConsistencyLoss(loss_type='mse')


        self.rampup_length = opt.get('weight_rampup', 30)
        self.consistency_weight_max = opt.get('consistency_weight', 1.0)
        self.warmup_epochs = opt.get('warmup_epochs', 30)
        self.teacher_start_epoch = self.warmup_epochs


        self.ldcp_start_epoch = int(opt.get('ldcp_start_epoch', 10 ** 9))
        print(f"[LDCP] start_epoch={self.ldcp_start_epoch}")

        self.depth_weights = opt.get('depth_weights', None)
        self.depth_encoder = opt.get('depth_encoder', 'vitl')
        self.depth_input_norm = opt.get('depth_input_norm', 'imagenet')
        self.ldcp_r = int(opt.get('ldcp_r', 3))
        self.ldcp_sigma = float(opt.get('ldcp_sigma', 0.5))
        self.ldcp_use_conf = bool(opt.get('ldcp_use_conf', True))
        self.ldcp_seg_weight = float(opt.get('ldcp_seg_weight', 0.5))
        self.ldcp_depth_minmax = bool(opt.get('ldcp_depth_minmax', True))
        self.eps = 1e-8

        self.lambda_route = float(opt.get('lambda_route', 0.2))
        self.lambda_feat = float(opt.get('lambda_feat', 0.5))
        self.lambda_cls_u = float(opt.get('lambda_cls_u', 1.0))
        self.lambda_reg_u = float(opt.get('lambda_reg_u', 1.0))
        self.lambda_seg_u = float(opt.get('lambda_seg_u', 0.5))

        self.lambda_res_keep = float(opt.get('lambda_res_keep', 0.05))
        self.res_keep_min = float(opt.get('res_keep_min', 0.01))

        self.depth_model = None
        if self.depth_weights is not None and len(str(self.depth_weights)) > 0:
            repo_root = self.opt.get(
                'depth_repo_root',
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_anything_v2'))
            )
            self.depth_model = DepthAnythingV2Wrapper(
                repo_root=repo_root,
                weights_path=self.depth_weights,
                encoder=self.depth_encoder,
                device=self.device,
                depth_input_norm=self.depth_input_norm,
                normalize_per_image=self.ldcp_depth_minmax
            )
            print(
                f"[LDCP] enabled. repo={repo_root}, r={self.ldcp_r}, sigma={self.ldcp_sigma}, use_conf={self.ldcp_use_conf}")
        else:
            print('[LDCP] disabled (depth_weights is None).')

    def compute_global_bias_mask(self, pre_cmap, gt_cmap):
        ""
        N, _, H, W = gt_cmap.size()
        err_cmap = (pre_cmap.detach() - gt_cmap.detach()).reshape(N, -1)
        im_err = err_cmap.sum(dim=1, keepdim=True)

        err_cmap = torch.sign(im_err) * err_cmap
        sorted_err, _ = torch.sort(err_cmap, dim=1, descending=False)
        sorted_err_accum = torch.cumsum(sorted_err, dim=1)

        thre_index = (sorted_err_accum <= 0).long().sum(dim=1)
        thre_index = torch.clamp(thre_index, max=err_cmap.size()[1] - 1)
        thre_value = torch.zeros_like(thre_index).float()
        mask = torch.zeros_like(err_cmap).detach()

        for ii in range(N):
            thre_value[ii] = sorted_err[ii, thre_index[ii]]
            if abs(im_err[ii, :]) > 0.5:
                mask[ii, :] = (err_cmap[ii, :] > thre_value[ii]).float()

        mask = mask.reshape(N, 1, H, W)
        return mask

    def _dgsa_keep_loss(self, aux):
        ""
        zero = torch.tensor(0.0, device=self.device)
        if not isinstance(aux, dict):
            return zero

        terms = []

        if 'fine_raw' in aux:
            fine_energy = aux['fine_raw'].abs().mean()
            terms.append(F.relu(self.res_keep_min - fine_energy))

        if 'coarse_raw' in aux:
            coarse_energy = aux['coarse_raw'].abs().mean()
            terms.append(F.relu(self.res_keep_min - coarse_energy))

        if 'dgsa_res' in aux:
            res_energy = aux['dgsa_res'].abs().mean()
            terms.append(F.relu(self.res_keep_min - res_energy))

        if len(terms) == 0:
            return zero

        return sum(terms) / float(len(terms))

    # -------------------- LDCP helpers --------------------
    def _normalize_depth_minmax(self, depth: torch.Tensor) -> torch.Tensor:
        ""
        B = depth.size(0)
        d = depth.reshape(B, -1)
        dmin = d.min(dim=1)[0].reshape(B, 1, 1, 1)
        dmax = d.max(dim=1)[0].reshape(B, 1, 1, 1)
        return (depth - dmin) / (dmax - dmin + 1e-8)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int = 14):
        ""
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)

        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (pad_h, pad_w)

    @torch.no_grad()
    def _infer_depth(self, rgb01: torch.Tensor, target_hw=None):
        x = rgb01
        # if self.depth_input_norm == 'imagenet':
        #     mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        #     std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        #     x = (x - mean) / std

        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, multiple=14)

        depth = self.depth_model(x_pad)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.size(1) != 1:
            depth = depth[:, :1, :, :]

        if pad_h > 0:
            depth = depth[:, :, :-pad_h, :]
        if pad_w > 0:
            depth = depth[:, :, :, :-pad_w]

        if target_hw is not None:
            depth = F.interpolate(depth, size=target_hw, mode='bilinear', align_corners=False)

        if self.ldcp_depth_minmax:
            depth = self._normalize_depth_minmax(depth)

        return depth

    def _compute_ldcp_weight(self, depth01: torch.Tensor, teacher_seg_prob: torch.Tensor) -> torch.Tensor:
        ""
        k = self.ldcp_r
        mean_local = F.avg_pool2d(depth01, kernel_size=2 * k + 1, stride=1, padding=k)
        delta = (depth01 - mean_local).abs()
        w_depth = torch.exp(-(delta * delta) / (self.ldcp_sigma * self.ldcp_sigma + 1e-12))

        W = w_depth
        if self.ldcp_use_conf:
            p_fg = teacher_seg_prob[:, 1:2, :, :]
            w_conf = torch.maximum(p_fg, 1.0 - p_fg)
            W = W * w_conf

        return W.clamp(0.0, 1.0)

    def _purify_teacher_seg(self, teacher_seg_prob: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        ""
        p_fg = teacher_seg_prob[:, 1:2, :, :]
        p_fg_tilde = (p_fg * W).clamp(0.0, 1.0)
        p_bg_tilde = 1.0 - p_fg_tilde
        return torch.cat([p_bg_tilde, p_fg_tilde], dim=1)

    def _weighted_l1_seg(self, student_prob: torch.Tensor, teacher_prob: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        ""
        diff = (student_prob - teacher_prob).abs().sum(dim=1, keepdim=True)  # [B,1,H,W]
        return (diff * W).sum() / (W.sum() + 1e-8)

    def train_epoch(self, train_loader, epoch):
        ""
        self.model.train()
        self.current_epoch = epoch


        if epoch == self.teacher_start_epoch:
            self.model.reset_teacher_from_student()
            print(f"[Teacher] reset from current student at epoch {epoch}")

        epoch_losses = {
            'total_loss': 0.0,
            'supervised_loss': 0.0,
            'reg_loss': 0.0,
            'cls_loss': 0.0,
            'seg_loss': 0.0,
            'unsupervised_loss': 0.0,
            'consistency_count': 0.0,
            'consistency_seg': 0.0,
            'consistency_route': 0.0,
            'consistency_feat': 0.0,
            'consistency_res_keep': 0.0
        }

        num_batches = 0
        psize = self.opt['W2DOpt']['DenPatchSize']

        for step, batch in enumerate(train_loader):
            self.model.global_step += 1


            weak_inputs = batch['weak_image'].to(self.device)
            strong_inputs = batch['strong_image'].to(self.device)
            density_maps = batch['density_map'].to(self.device)
            seg_maps = batch['seg_map'].to(self.device)
            is_labeled = batch['is_labeled']

            N = weak_inputs.size(0)
            labeled_mask = is_labeled
            unlabeled_mask = ~is_labeled
            N_l = labeled_mask.sum().item()
            N_u = unlabeled_mask.sum().item()

            self.optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            unsupervised_loss = torch.tensor(0.0, device=self.device)


            if N_l > 0:
                labeled_inputs = weak_inputs[labeled_mask]
                labeled_density = density_maps[labeled_mask]
                labeled_seg = seg_maps[labeled_mask]


                gt_cmap = get_local_count(labeled_density, psize, psize)


                #     print("[DBG] gt_cmap shape:", tuple(gt_cmap.shape),
                #           "min/mean/max/sum:",
                #           gt_cmap.min().item(),
                #           gt_cmap.mean().item(),
                #           gt_cmap.max().item(),
                #           gt_cmap.sum().item())


                class_targets = self.model.encode_count(gt_cmap).squeeze(1)


                # Cmap, pred_seg = self.model(labeled_inputs, use_teacher=False)
                Cmap, pred_seg, s_aux = self.model(
                    labeled_inputs,
                    use_teacher=False,
                    depth=None,
                    return_aux=True
                )

                # if step < 3:
                #     print("[DBG] Cmap shape:", tuple(Cmap.shape),
                #           "min/mean/max/sum:",
                #           Cmap.min().item(),
                #           Cmap.mean().item(),
                #           Cmap.max().item(),
                #           Cmap.sum().item())


                cls_loss = torch.tensor(0.0, device=self.device)
                if self.loss_dict.get('cls_loss') is not None:
                    pre_cls = self.model.encode_count(Cmap.detach()).squeeze(1)
                    mask = (pre_cls.squeeze() != class_targets.detach().squeeze()).float()

                    # if step < 3:
                    #     ratio = (mask > 0).float().mean().item()
                    #     print("[DBG] mask_cls ratio:", ratio, "mask_sum:", mask.sum().item())

                    cls_loss = ((gt_cmap - Cmap).squeeze().abs() * mask.squeeze()).sum() / \
                               torch.clamp(mask.sum(), min=1.0)


                reg_loss = torch.tensor(0.0, device=self.device)
                if self.loss_dict.get('total_reg_loss') is not None:
                    mask = self.compute_global_bias_mask(Cmap, gt_cmap)
                    reg_loss = ((Cmap - gt_cmap).abs() * mask).sum() / torch.clamp(mask.sum(), min=1.0)

                    # if step < 3:
                    #     ratio = (mask > 0).float().mean().item()
                    #     print("[DBG] mask_reg ratio:", ratio, "mask_sum:", mask.sum().item())


                seg_loss = torch.tensor(0.0, device=self.device)
                if self.loss_dict.get('loss_seg') is not None and pred_seg is not None:
                    seg_loss = loss_seg(pred_seg, labeled_seg)


                res_keep_loss_sup = torch.tensor(0.0, device=self.device)
                if self.lambda_res_keep > 0:
                    res_keep_loss_sup = self._dgsa_keep_loss(s_aux)

                supervised_loss = (
                        self.loss_w.get('cls_loss', 1.0) * cls_loss +
                        self.loss_w.get('total_reg_loss', 1.0) * reg_loss +
                        self.loss_w.get('loss_seg', 0.5) * seg_loss +
                        self.lambda_res_keep * res_keep_loss_sup
                )

                total_loss = total_loss + supervised_loss

                epoch_losses['supervised_loss'] += supervised_loss.item()
                epoch_losses['reg_loss'] += reg_loss.item()
                epoch_losses['cls_loss'] += cls_loss.item()
                epoch_losses['seg_loss'] += seg_loss.item()
                epoch_losses['consistency_res_keep'] += res_keep_loss_sup.item()


            if N_u > 0 and epoch >= self.warmup_epochs:
                unlabeled_weak = weak_inputs[unlabeled_mask]
                unlabeled_strong = strong_inputs[unlabeled_mask]

                weak_raw = batch.get('weak_image_raw', None)
                if weak_raw is None:
                    raise RuntimeError("DGSA/LDCP 需要 batch['weak_image_raw']")
                unlabeled_weak_raw = weak_raw.to(self.device)[unlabeled_mask]

                # 1) teacher depth
                with torch.no_grad():
                    depth_full = self._infer_depth(unlabeled_weak_raw, target_hw=None)

                # 2) student forward (RGB only)
                u_s_Cmap, u_s_seg, u_s_aux = self.model(
                    unlabeled_strong,
                    use_teacher=False,
                    depth=None,
                    return_aux=True
                )

                # 3) teacher forward (RGB + depth)
                with torch.no_grad():
                    u_t_Cmap, u_t_seg, u_t_aux = self.model(
                        unlabeled_weak,
                        use_teacher=True,
                        depth=depth_full,
                        return_aux=True
                    )

                    u_t_seg = u_t_seg.detach()
                    u_t_Cmap = u_t_Cmap.detach()
                    u_t_route = u_t_aux['route_prob'].detach()
                    u_t_res = u_t_aux['dgsa_res'].detach()


                    u_t_cls_targets = self.model.encode_count(u_t_Cmap, use_teacher=True).squeeze(1).detach()


                cls_consistency = torch.tensor(0.0, device=self.device)
                u_s_cls = self.model.encode_count(u_s_Cmap.detach()).squeeze(1)
                mask_cls = (u_s_cls.squeeze() != u_t_cls_targets.squeeze()).float()
                cls_consistency = ((u_t_Cmap - u_s_Cmap).squeeze().abs() * mask_cls.squeeze()).sum() / \
                                  torch.clamp(mask_cls.sum(), min=1.0)


                reg_consistency = torch.tensor(0.0, device=self.device)
                mask_reg = self.compute_global_bias_mask(u_s_Cmap, u_t_Cmap)
                reg_consistency = ((u_s_Cmap - u_t_Cmap).abs() * mask_reg).sum() / \
                                  torch.clamp(mask_reg.sum(), min=1.0)


                seg_consistency = torch.tensor(0.0, device=self.device)
                use_ldcp = (epoch >= self.ldcp_start_epoch) and (self.depth_model is not None)

                if (u_s_seg is not None) and (u_t_seg is not None):
                    if use_ldcp:
                        depth_seg = F.interpolate(
                            depth_full,
                            size=u_t_seg.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        W = self._compute_ldcp_weight(depth_seg, u_t_seg)
                        u_t_seg_pur = self._purify_teacher_seg(u_t_seg, W)
                        seg_consistency = self._weighted_l1_seg(u_s_seg, u_t_seg_pur, W)
                    else:
                        seg_consistency = F.mse_loss(u_s_seg, u_t_seg)


                route_consistency = F.l1_loss(
                    u_s_aux['route_prob'],
                    u_t_route
                )


                feat_consistency = F.mse_loss(
                    u_s_aux['dgsa_res'],
                    u_t_res
                )


                res_keep_loss_unsup = torch.tensor(0.0, device=self.device)
                if self.lambda_res_keep > 0:
                    res_keep_loss_unsup = self._dgsa_keep_loss(u_s_aux)

                unsupervised_loss = (
                        self.lambda_cls_u * cls_consistency +
                        self.lambda_reg_u * reg_consistency +
                        self.lambda_seg_u * seg_consistency +
                        self.lambda_route * route_consistency +
                        self.lambda_feat * feat_consistency +
                        self.lambda_res_keep * res_keep_loss_unsup
                )

                rampup_epoch = epoch - self.warmup_epochs
                consistency_weight = exp_rampup(rampup_epoch, self.rampup_length) * self.consistency_weight_max
                total_loss = total_loss + consistency_weight * unsupervised_loss

                epoch_losses['unsupervised_loss'] += unsupervised_loss.item()
                epoch_losses['consistency_count'] += (cls_consistency + reg_consistency).item()
                epoch_losses['consistency_seg'] += seg_consistency.item()
                epoch_losses['consistency_route'] += route_consistency.item()
                epoch_losses['consistency_feat'] += feat_consistency.item()
                epoch_losses['consistency_res_keep'] += res_keep_loss_unsup.item()


            total_loss.backward()


            torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), 20)

            self.optimizer.step()



            if epoch >= self.teacher_start_epoch:
                self.model.update_teacher_model()


            if step % 10 == 0:
                print(f'Epoch: {epoch}, Step: {step}/{len(train_loader)}, '
                      f'Total: {total_loss.item():.4f}, '
                      f'Sup: {epoch_losses["supervised_loss"] / max(1, num_batches + 1):.4f}, '
                      f'Unsup: {epoch_losses["unsupervised_loss"] / max(1, num_batches + 1):.4f}')

            epoch_losses['total_loss'] += total_loss.item()
            num_batches += 1

        for k in epoch_losses.keys():
            epoch_losses[k] /= max(1, num_batches)

        return epoch_losses
