# -*- coding: utf-8 -*-
""

import os
import sys
import argparse
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time


CUR_DIR = os.path.split(os.path.abspath(__file__))[0]
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

PROJ_ROOT = os.path.dirname(CUR_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


from utils.IOtools import txt_write, get_config_str
from Network.search_template import get_template_fromdset
from cls_funcs.Interval_paritition import interval_divide
from Network.class_func import get_local_count

from Network.mean_teacher_model import (
    SemiSupervisedCountingModel,
    SemiSupervisedTrainer,
    exp_rampup
)

from utils.semi_supervised_dataset import (
    create_semi_supervised_dataloaders,
    ValidationDataset
)

from utils.visualization import SegmentationVisualizer, visualize_segmentation_predictions


def setup_seed(seed):
    ""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args():
    ""
    parser = argparse.ArgumentParser(
        description='Semi-Supervised Wheat Ears Counting with Mean Teacher + LDCP'
    )


    parser.add_argument('--dataroot', type=str, default=os.path.join(CUR_DIR, 'data', 'GWHD2021'),
                        help='数据集根目录')
    parser.add_argument('--sdset', type=str, default='GWHD2021',
                        help='数据集名称', choices=['JHU', 'GWHD2021', 'WHDSC2', 'MS'])
    parser.add_argument('--crop-size', type=int, default=512,
                        help='训练图像裁剪尺寸')
    parser.add_argument('--labeled-list', type=str, default=os.path.join(CUR_DIR, 'data', 'GWHD2021', 'GWHD40.txt'),
                        help='预定义的有标签数据列表文件路径')


    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--epoch-num', type=int, default=300, help='训练轮数')
    parser.add_argument('--bs', type=int, default=16, help='批次大小')
    parser.add_argument('--num-labeled', type=int, default=8, help='每个批次中有标签样本数量')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--milestones', default=[2500], type=int, nargs='+',
                        help='学习率衰减的epoch节点')


    parser.add_argument('--ema-decay', type=float, default=0.99, help='教师模型EMA更新系数')
    parser.add_argument('--labeled-ratio', type=float, default=0.05, help='有标签数据比例')
    parser.add_argument('--warmup-epochs', type=int, default=30, help='纯监督训练的预热轮数')
    parser.add_argument('--weight-rampup', type=int, default=20, help='一致性损失权重增长轮数')
    parser.add_argument('--consistency-weight', type=float, default=1.0, help='一致性损失最大权重')


    parser.add_argument('--ldcp-start-epoch', type=int, default=150,
                        help='LDCP 从第几轮开始启用（例如 200）。为空则默认等于 warmup-epochs。')


    parser.add_argument('--ptype', type=str, default='log',
                        help='区间划分类型', choices=['log', 'linear'])


    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(CUR_DIR, 'outputs'),
                        help='模型和日志保存目录')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--val-epoch', type=int, default=1, help='验证间隔轮数')
    parser.add_argument('--start-val', type=int, default=5, help='开始验证的轮数')
    parser.add_argument('--test-only', action='store_true', default='', help='仅测试模式')
    parser.add_argument('--vis-epoch', type=int, default=30, help='可视化间隔轮数')
    parser.add_argument('--vis-samples', type=int, default=20, help='每次可视化的样本数')


    parser.add_argument('--depth-weights', type=str, default=os.path.join(CUR_DIR, 'checkpoints', 'depth_anything_v2_vitl.pth'),
                        help='DepthAnythingV2 本地权重路径，例如 /path/depth_anything_v2_vitl.pth')
    parser.add_argument('--depth-encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl'],
                        help='DepthAnythingV2 encoder 类型（要与权重匹配）')
    parser.add_argument('--depth-input-norm', type=str, default='imagenet',
                        choices=['imagenet', 'none'],
                        help='喂给深度模型前是否做 ImageNet 归一化')
    parser.add_argument('--ldcp-r', type=int, default=1, help='LDCP 邻域半径 r（建议 3 或 5）')
    parser.add_argument('--ldcp-sigma', type=float, default=0.6, help='LDCP sigma_l（建议 0.3~1.0）')
    parser.add_argument('--ldcp-use-conf', type=int, default=0, choices=[0, 1],
                        help='是否融合 teacher 置信度 W_conf')
    parser.add_argument('--ldcp-seg-weight', type=float, default=0.5,
                        help='无监督 seg 一致性内部权重（在 mean_teacher_model 里乘）')
    parser.add_argument('--ldcp-depth-minmax', type=int, default=1, choices=[0, 1],
                        help='是否对每张图深度做 min-max 到 [0,1]（强烈建议=1）')


    parser.add_argument('--dgsa-dilation', type=int, default=2,
                        help='DGSA coarse branch 的空洞卷积膨胀率')

    parser.add_argument('--lambda-route', type=float, default=0.05,
                        help='route distillation 权重')


    parser.add_argument('--lambda-feat', type=float, default=0.05,
                        help='residual distillation 权重')


    parser.add_argument('--lambda-res-keep', type=float, default=0,
                        help='DGSA 防塌缩约束权重')
    parser.add_argument('--res-keep-min', type=float, default=0.03,
                        help='DGSA 分支/残差平均绝对激活的最小阈值')

    parser.add_argument('--dgsa-alpha-init', type=float, default=0.2,
                        help='DGSA residual 融合初始 alpha')
    parser.add_argument('--dgsa-alpha-max', type=float, default=1,
                        help='DGSA residual 融合 alpha 的最大值')


    parser.add_argument('--lambda-cls-u', type=float, default=1.0,
                        help='无监督 cls consistency 权重')
    parser.add_argument('--lambda-reg-u', type=float, default=1.0,
                        help='无监督 reg consistency 权重')
    parser.add_argument('--lambda-seg-u', type=float, default=0.5,
                        help='无监督 seg consistency 权重')

    return parser.parse_args()


class SemiSupervisedCountingTrainer:
    ""

    def __init__(self, args):
        self.args = args
        self._setup()

    def _setup(self):
        args = self.args

        # seed
        if args.seed != -1:
            setup_seed(args.seed)
            print(f'Random seed set to {args.seed}')

        # GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f'Using GPU: {args.device}')
        else:
            raise RuntimeError('CUDA is not available')


        DROOT_list = {
            'JHU': '',
            'GWHD2021': '',
            'WHDSC2': '',
            'MS': ''
        }
        self.sdroot = os.path.join(args.dataroot, DROOT_list[args.sdset])


        sub_dir = (f"semi_{args.sdset}_lr{args.lr}_bs{args.bs}_"
                   f"labeled{args.labeled_ratio}_ema{args.ema_decay}")
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # log
        self.log_path = os.path.join(self.save_dir, 'train.log')
        self._log_config()

        # vis
        vis_save_dir = os.path.join(self.save_dir, 'visualizations')
        self.visualizer = SegmentationVisualizer(vis_save_dir)
        self._log(f"可视化将保存到: {vis_save_dir}")

        # opt
        self.opt = self._create_opt_dict()

        # loss config
        self.loss_dict, self.loss_w = self._create_loss_config()

        # class indice
        self._setup_class_indices()

        # model
        self._create_model()

        # dataloaders
        self._create_dataloaders()

        # optimizer/scheduler
        self._create_optimizer()

        # trainer
        self.trainer = SemiSupervisedTrainer(
            model=self.model,
            optimizer=self.optimizer,
            opt=self.opt,
            loss_dict=self.loss_dict,
            loss_w=self.loss_w,
            device=self.device
        )

        # state
        self.start_epoch = 0
        self.best_mae = float('inf')
        self.best_mse = float('inf')
        self.train_losses = []
        self.val_maes = []

        # resume
        if args.resume:
            self._load_checkpoint(args.resume)

    def _log_config(self):
        config_msg = f"\n{'=' * 60}\n"
        config_msg += f"半监督训练开始 - {datetime.datetime.now()}\n"
        config_msg += f"{'=' * 60}\n"
        for k, v in self.args.__dict__.items():
            config_msg += f"{k}: {v}\n"
        config_msg += f"{'=' * 60}\n"

        txt_write(self.log_path, config_msg, mode='w')
        print(config_msg)

    def _create_opt_dict(self):
        args = self.args


        EmbedOpt = {
            'block_num': 5,
            'decode_num': 0,
            'load_weights': True,
            'IF_VGG16bn': True,
            'IF_freeze_bn': False
        }

        W2DOpt = {
            'DenPatchSize': 8,
            'EPS': 1e-8
        }


        ldcp_start_epoch = args.ldcp_start_epoch
        if ldcp_start_epoch is None:
            ldcp_start_epoch = args.warmup_epochs

        return {
            'sdroot': self.sdroot,
            'crop_size': args.crop_size,
            'bs': args.bs,
            'num_workers': args.num_workers,
            'save_folder': self.save_dir,
            'EmbedOpt': EmbedOpt,
            'W2DOpt': W2DOpt,

            'warmup_epochs': args.warmup_epochs,
            'weight_rampup': args.weight_rampup,
            'consistency_weight': args.consistency_weight,

            'ldcp_start_epoch': args.ldcp_start_epoch,

            'depth_repo_root': os.path.join(CUR_DIR, 'models', 'depth_anything_v2'),
            'depth_weights': args.depth_weights,
            'depth_encoder': args.depth_encoder,
            'depth_input_norm': args.depth_input_norm,
            'ldcp_r': args.ldcp_r,
            'ldcp_sigma': args.ldcp_sigma,
            'ldcp_use_conf': bool(args.ldcp_use_conf),
            'ldcp_seg_weight': args.ldcp_seg_weight,
            'ldcp_depth_minmax': bool(args.ldcp_depth_minmax),

            'dgsa_dilation': args.dgsa_dilation,
            'dgsa_alpha_init': args.dgsa_alpha_init,
            'dgsa_alpha_max': args.dgsa_alpha_max,
            'lambda_route': args.lambda_route,
            'lambda_feat': args.lambda_feat,
            'lambda_cls_u': args.lambda_cls_u,
            'lambda_reg_u': args.lambda_reg_u,
            'lambda_seg_u': args.lambda_seg_u,
            'lambda_res_keep': args.lambda_res_keep,
            'res_keep_min': args.res_keep_min,
        }


    def _create_loss_config(self):
        loss_dict = {
            'total_reg_loss': nn.L1Loss(reduction='mean'),
            'cls_loss': nn.L1Loss(reduction='mean'),
            'loss_seg': 'loss_seg'
        }
        loss_w = {
            'total_reg_loss': 1.0,
            'cls_loss': 1.0,
            'loss_seg': 0.5
        }
        return loss_dict, loss_w

    def _setup_class_indices(self):
        args = self.args
        root = os.path.join(self.sdroot, 'train')

        choose_t_func = get_template_fromdset(
            root,
            psize=self.opt['W2DOpt']['DenPatchSize'],
            save_folder=os.path.join(CUR_DIR, f'TemplateTongJi-{args.sdset}')
        )
        choose_t_func.compute_local_count()

        patch_count_gt = choose_t_func.ret_patch_gt()
        Cmax = patch_count_gt.max().item()

        self.class_indice, self.class2count_spec = interval_divide(
            patch_count_gt,
            vmin=5e-3,
            vmax=Cmax,
            num=100,
            partition=args.ptype
        )

    def _create_model(self):

        self.opt['class_indice'] = self.class_indice
        self.opt['class2count_spec'] = self.class2count_spec
        self.opt['ema_decay'] = self.args.ema_decay


        self.model = SemiSupervisedCountingModel(self.opt, self.loss_dict, self.loss_w).to(self.device)

    def _create_dataloaders(self):
        args = self.args
        self.train_loader, self.val_loader = create_semi_supervised_dataloaders(
            self.opt,
            labeled_ratio=args.labeled_ratio,
            num_labeled_per_batch=args.num_labeled,
            labeled_list_path=args.labeled_list
        )
        self._log(f'Number of training batches: {len(self.train_loader)}')
        self._log(f'Number of validation samples: {len(self.val_loader)}')

    def _create_optimizer(self):
        args = self.args
        self.optimizer = optim.Adam(
            self.model.student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.milestones,
            gamma=0.1
        )

    def _log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_msg = f"[{timestamp}] {level}: {message}\n"
        txt_write(self.log_path, full_msg, mode='a')
        print(f"[{level}] {message}")

    def _load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.student.load_state_dict(checkpoint['student_state_dict'])
            if 'teacher_state_dict' in checkpoint:
                self.model.teacher.load_state_dict(checkpoint['teacher_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_mae = checkpoint.get('best_mae', float('inf'))
            self.best_mse = checkpoint.get('best_mse', float('inf'))
            self.model.global_step = checkpoint.get('global_step', 0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_maes = checkpoint.get('val_maes', [])

            self._log(f'Resumed from epoch {self.start_epoch}')
        else:
            self._log(f'No checkpoint found at {checkpoint_path}')

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.model.student.state_dict(),
            'teacher_state_dict': self.model.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.model.global_step,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'train_losses': self.train_losses,
            'val_maes': self.val_maes
        }

        if epoch % 10 == 0:
            save_path = os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save(checkpoint, save_path)

        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        if is_best:
            best_student_path = os.path.join(self.save_dir, 'best_student.pth')
            torch.save(self.model.student.state_dict(), best_student_path)

            best_teacher_path = os.path.join(self.save_dir, 'best_teacher.pth')
            torch.save(self.model.teacher.state_dict(), best_teacher_path)

            self._log(f"Saved best models: MAE={self.best_mae:.2f}, MSE={self.best_mse:.2f}")

    def _plot_learning_curve(self):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            if len(self.train_losses) > 0:
                axes[0].plot(self.train_losses, 'b-', label='Train Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss')
                axes[0].legend()
                axes[0].grid(True)

            if len(self.val_maes) > 0:
                val_epochs = [self.args.start_val + i * self.args.val_epoch
                              for i in range(len(self.val_maes))]
                axes[1].plot(val_epochs, self.val_maes, 'r-', label='Val MAE')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].set_title('Validation MAE')
                axes[1].legend()
                axes[1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'learning_curve.png'), dpi=150)
            plt.close()

        except Exception as e:
            self._log(f"Failed to plot learning curve: {e}", level="WARNING")

    def visualize(self, epoch, num_samples=5):
        try:
            visualize_segmentation_predictions(
                net=self.model,
                dataloader=self.val_loader,
                visualizer=self.visualizer,
                epoch=epoch,
                device=self.device,
                num_samples=num_samples
            )
            self._log(f"可视化完成，已保存到: {self.visualizer.save_dir}/epoch_{epoch}/")
        except Exception as e:
            self._log(f"可视化出错: {e}", level="WARNING")
            import traceback
            traceback.print_exc()

    def validate(self):
        self.model.eval()
        # self.model.train()

        if hasattr(self.model.student, 'debug_forward_count'):
            self.model.student.debug_forward_count = 0
        if hasattr(self.model.teacher, 'debug_forward_count'):
            self.model.teacher.debug_forward_count = 0

        epoch_res = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                inputs = batch['image'].to(self.device)
                gt_counts = batch['all_num']

                outputs, _ = self.model(inputs, use_teacher=False)
                pred_count = outputs.sum().item()
                gt_count = gt_counts[0].item()


                #     print("[VAL] gt_count:", gt_count, "pred_count:", pred_count)
                #     print("[VAL] Cmap stats:",
                #           outputs.min().item(),
                #           outputs.mean().item(),
                #           outputs.max().item(),
                #           outputs.sum().item())

                res = gt_count - pred_count
                epoch_res.append(res)



                if i % 20 == 0:
                    print(f'Val: [{i + 1}/{len(self.val_loader)}] '
                          f'Pred: {pred_count:.2f}, GT: {gt_count:.2f}, '
                          f'Error: {res:.2f}')

        epoch_res = np.array(epoch_res)
        mae = np.mean(np.abs(epoch_res))
        mse = np.sqrt(np.mean(np.square(epoch_res)))

        self.model.train()
        return {'mae': mae, 'mse': mse}

    def train(self):
        args = self.args
        self._log(f"Starting training from epoch {self.start_epoch}")

        for epoch in range(self.start_epoch, args.epoch_num):
            self._log(f"Epoch {epoch}/{args.epoch_num - 1}")

            train_losses = self.trainer.train_epoch(self.train_loader, epoch)
            self.train_losses.append(train_losses['total_loss'])

            self.scheduler.step()

            loss_msg = (f"Train Loss: {train_losses['total_loss']:.4f}, "
                        f"Sup: {train_losses['supervised_loss']:.4f}, "
                        f"Unsup: {train_losses['unsupervised_loss']:.4f}")
            self._log(loss_msg)

            if epoch >= args.start_val and epoch % args.val_epoch == 0:
                val_results = self.validate()
                mae, mse = val_results['mae'], val_results['mse']
                self.val_maes.append(mae)

                is_best = (mae + mse) < (self.best_mae + self.best_mse)
                if is_best:
                    self.best_mae = mae
                    self.best_mse = mse

                self._save_checkpoint(epoch, is_best)

                self._log(f"Val MAE: {mae:.2f}, MSE: {mse:.2f}")
                self._log(f"Best MAE: {self.best_mae:.2f}, Best MSE: {self.best_mse:.2f}")

                self._plot_learning_curve()

                if epoch % args.vis_epoch == 0:
                    self.visualize(epoch, num_samples=args.vis_samples)
            else:
                self._save_checkpoint(epoch, False)

        self._log("Training completed!")
        self._log(f"Best Results - MAE: {self.best_mae:.2f}, MSE: {self.best_mse:.2f}")

    def test(self):
        ""

        best_student_path = os.path.join(self.save_dir, 'best_student.pth')
        if os.path.exists(best_student_path):
            self.model.student.load_state_dict(
                torch.load(best_student_path, map_location=self.device)
            )
            self._log(f'Loaded best student model from {best_student_path}')
        else:

            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                self.model.student.load_state_dict(
                    torch.load(best_model_path, map_location=self.device)
                )
                self._log(f'Loaded best model from {best_model_path}')


        test_root = os.path.join(self.sdroot, 'test')
        if not os.path.exists(test_root):
            self._log(f"Test directory not found: {test_root}", level="ERROR")
            return None

        test_dataset = ValidationDataset(
            root=test_root,
            DIV=self.opt['W2DOpt']['DenPatchSize']
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=self.args.num_workers)

        self._log(f"Testing on {len(test_dataset)} images from {test_root}")


        self.model.eval()
        epoch_res = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                inputs = batch['image'].to(self.device)
                gt_counts = batch['all_num']
                name = batch['name'][0] if isinstance(batch['name'], (list, tuple)) else batch['name']

                outputs, _ = self.model(inputs, use_teacher=False)
                pred_count = outputs.sum().item()
                gt_count = gt_counts[0].item()

                res = gt_count - pred_count
                epoch_res.append(res)

                print(f'Test: [{i + 1}/{len(test_loader)}] {name}: '
                      f'Pred: {pred_count:.2f}, GT: {gt_count:.2f}, Error: {res:.2f}')

        epoch_res = np.array(epoch_res)
        mae = np.mean(np.abs(epoch_res))
        mse = np.sqrt(np.mean(np.square(epoch_res)))

        self._log(f"Test Results - MAE: {mae:.2f}, MSE: {mse:.2f}")


        result_path = os.path.join(self.save_dir, 'test_results.txt')
        with open(result_path, 'w') as f:
            f.write(f'Test Results:\n')
            f.write(f'MAE: {mae:.2f}\n')
            f.write(f'MSE: {mse:.2f}\n')


        self._log("Generating test visualizations...")
        try:
            visualize_segmentation_predictions(
                net=self.model,
                dataloader=test_loader,
                visualizer=self.visualizer,
                epoch='test',
                device=self.device,
                num_samples=10
            )
        except Exception as e:
            self._log(f"Test visualization error: {e}", level="WARNING")

        return {'mae': mae, 'mse': mse}



def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    trainer = SemiSupervisedCountingTrainer(args)

    if args.test_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    start_time = time()
    main()

    total_time = time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f'\nTotal time: {hours}h {minutes}m {seconds}s')
