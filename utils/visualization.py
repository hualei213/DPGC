# utils/visualization.py
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class SegmentationVisualizer:
    ""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.rgb_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.rgb_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def denormalize_image(self, image):
        ""
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = image_np * self.rgb_std + self.rgb_mean
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        return image_np

    def process_segmentation_mask(self, seg_pred, target_size=None):
        ""
        if isinstance(seg_pred, torch.Tensor):
            seg_pred = seg_pred.cpu()


        if seg_pred.size(1) == 2:
            foreground_prob = seg_pred[0, 1, :, :].numpy()
        else:
            foreground_prob = seg_pred[0, 0, :, :].numpy()

        if target_size is not None:
            foreground_prob = cv2.resize(foreground_prob, target_size, interpolation=cv2.INTER_LINEAR)

        return foreground_prob

    def create_colored_mask(self, mask, colormap='jet'):
        ""

        if mask.max() == mask.min():

            mask_norm = np.zeros_like(mask)
        else:
            mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        if colormap == 'jet':
            colored_mask = cm.jet(mask_norm)[:, :, :3]
        elif colormap == 'viridis':
            colored_mask = cm.viridis(mask_norm)[:, :, :3]
        elif colormap == 'hot':
            colored_mask = cm.hot(mask_norm)[:, :, :3]
        else:
            colored_mask = cm.viridis(mask_norm)[:, :, :3]

        colored_mask = (colored_mask * 255).astype(np.uint8)
        return colored_mask

    def overlay_mask_on_image(self, image, mask, alpha=0.5):
        ""
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
        return overlay

    def save_segmentation_visualization(self, image, seg_pred, image_name, epoch, batch_idx):
        ""
        epoch_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)


        original_image = self.denormalize_image(image[0])
        H, W = original_image.shape[:2]

        try:

            mask = self.process_segmentation_mask(seg_pred, (W, H))

            print(f"[DEBUG] {image_name}: mask min={mask.min():.4f}, max={mask.max():.4f}")


            colored_mask = self.create_colored_mask(mask, colormap='viridis')
            mask_filename = f'{image_name}_batch{batch_idx}_mask.png'
            mask_path = os.path.join(epoch_dir, mask_filename)
            cv2.imwrite(mask_path, colored_mask)



            binary_mask = (mask > 0.5).astype(np.uint8) * 255


            colored_binary = np.zeros((H, W, 3), dtype=np.uint8)
            colored_binary[binary_mask > 0] = [0, 255, 0]


            overlay_binary = cv2.addWeighted(original_image, 0.6, colored_binary, 0.4, 0)

            overlay_binary_filename = f'{image_name}_batch{batch_idx}_binary_overlay.png'
            overlay_binary_path = os.path.join(epoch_dir, overlay_binary_filename)
            cv2.imwrite(overlay_binary_path, overlay_binary)



            binary_mask_uint8 = binary_mask.astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            overlay_contour = original_image.copy()
            cv2.drawContours(overlay_contour, contours, -1, (0, 255, 0), 2)

            overlay_contour_filename = f'{image_name}_batch{batch_idx}_contour_overlay.png'
            overlay_contour_path = os.path.join(epoch_dir, overlay_contour_filename)
            cv2.imwrite(overlay_contour_path, overlay_contour)


            colored_mask_enhanced = self.create_colored_mask(mask, colormap='viridis')
            overlay_heatmap = cv2.addWeighted(original_image, 0.5, colored_mask_enhanced, 0.5, 0)

            overlay_heatmap_filename = f'{image_name}_batch{batch_idx}_heatmap_overlay.png'
            overlay_heatmap_path = os.path.join(epoch_dir, overlay_heatmap_filename)
            cv2.imwrite(overlay_heatmap_path, overlay_heatmap)

        except Exception as e:
            print(f"处理 {image_name} 时出错: {e}")

        print(f"已保存可视化: {image_name} at epoch {epoch}")

    def save_gt_segmentation_visualization(self, image, gt_seg_map, image_name, epoch, batch_idx):
        ""
        epoch_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)


        original_image = self.denormalize_image(image[0])
        H, W = original_image.shape[:2]


        gt_seg = gt_seg_map[0, 0].cpu().numpy()


        gt_seg_resized = cv2.resize(gt_seg, (W, H), interpolation=cv2.INTER_NEAREST)


        gt_colored_mask = self.create_colored_mask(gt_seg_resized, colormap='viridis')


        gt_overlay = self.overlay_mask_on_image(original_image, gt_colored_mask, alpha=0.4)


        gt_mask_filename = f'{image_name}_batch{batch_idx}_GT_mask.png'
        gt_mask_path = os.path.join(epoch_dir, gt_mask_filename)
        cv2.imwrite(gt_mask_path, gt_colored_mask)


        gt_overlay_filename = f'{image_name}_batch{batch_idx}_GT_overlay.png'
        gt_overlay_path = os.path.join(epoch_dir, gt_overlay_filename)
        cv2.imwrite(gt_overlay_path, gt_overlay)

        print(f"已保存GT可视化: {image_name} at epoch {epoch}")


def visualize_segmentation_predictions(net, dataloader, visualizer, epoch, device, num_samples=3):
    ""
    net.eval()
    print(f"\n=== 正在生成epoch {epoch}的分割可视化 ===")

    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            if batch_idx >= num_samples:
                break

            inputs = sample['image'].to(device)
            names = sample['name']

            try:

                _, seg_pred = net(inputs)


                if len(names) > 0:

                    if isinstance(names, list):
                        image_name = names[0]
                    elif isinstance(names, tuple):
                        image_name = names[0]
                    else:
                        image_name = str(names[0]) if hasattr(names, '__getitem__') else str(names)


                    image_name = str(image_name).replace('/', '_').replace('\\', '_')


                    visualizer.save_segmentation_visualization(
                        inputs[0:1],
                        seg_pred[0:1],
                        image_name,
                        epoch,
                        batch_idx
                    )


                    if 'seg_map' in sample:
                        visualizer.save_gt_segmentation_visualization(
                            inputs[0:1],
                            sample['seg_map'][0:1],
                            image_name,
                            epoch,
                            batch_idx
                        )

            except Exception as e:
                print(f"处理batch {batch_idx}时出错: {e}")
                continue

    net.train()
    print(f"分割可视化已保存到: {visualizer.save_dir}/epoch_{epoch}/")



def test_visualization():
    ""
    print("开始测试可视化功能...")

    save_dir = "test_visualizations"
    visualizer = SegmentationVisualizer(save_dir)


    dummy_image = torch.randn(1, 3, 512, 512)
    dummy_seg_pred = torch.randn(1, 2, 512, 512)
    dummy_seg_pred = F.softmax(dummy_seg_pred, dim=1)

    try:
        visualizer.save_segmentation_visualization(
            dummy_image, dummy_seg_pred, "test_image", epoch=50, batch_idx=0
        )
        print("测试成功！检查 test_visualizations 文件夹")
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    test_visualization()
