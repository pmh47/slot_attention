import os
import numpy as np
from absl import app
from absl import flags
from tqdm import tqdm
from collections import defaultdict

import tensorflow as tf
import torch
import torch.nn.functional as F

import slot_attention.model as model_utils

# The following are from o3d-nerf, which should therefore be on PYTHONPATH:
import evaluation.metrics as metrics
from model.loss import img2mse, mse2psnr


FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", "/root/workspace/data/arrow", "Root folder for the dataset.")
flags.DEFINE_string("split", "train", "Which train/test/ood* split to use.")
flags.DEFINE_string("ckpt_path", "/root/workspace/slot_attention/checkpoints/arrow_2022-04-19", "Where to load the checkpoint from.")
flags.DEFINE_string("out_path", "", "Where to write reconstruction/mask images.")
flags.DEFINE_string("metrics_filename", "/tmp/statistics.txt", "Filename to write final metric values as text.")
flags.DEFINE_integer("resolution", 96, "Image resolution")
flags.DEFINE_integer("num_slots", 5, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_integer("num_scenes", 100, "Number of scenes to evaluate.")


def load_model(checkpoint_dir, num_slots=11, num_iters=3, batch_size=16):
    model = model_utils.build_model(
        (FLAGS.resolution, FLAGS.resolution), batch_size, num_slots, num_iters,
        model_type="object_discovery")

    ckpt = tf.train.Checkpoint(network=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=checkpoint_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from", ckpt_manager.latest_checkpoint)
    else:
        print("Initialised model from scratch")

    return model


def get_prediction(model, image):

    def renormalize(x):
        """Renormalize from [-1, 1] to [0, 1]."""
        return x / 2. + 0.5

    recon_combined, recons, masks, slots = model(image * 2. - 1.)
    recon_combined = renormalize(recon_combined)
    recons = renormalize(recons)

    return recon_combined, recons, masks, slots


def load_gqn_masks(scene_index, frame_indices):

    def convert_masks(gt_masks):
        # Duplicated from o3d-nerf / data / gqn
        # returns [N_frames, H, W, N_objects + 1] where [..., 0] is background
        gt_masks_int = gt_masks[..., 0] + gt_masks[..., 1] * 256 + gt_masks[..., 2] * 256 * 256
        gt_masks_int = torch.Tensor(gt_masks_int.astype(np.int32))
        output, inverse_indices = torch.unique(gt_masks_int, sorted=True, return_inverse=True)
        return F.one_hot(inverse_indices)

    rgb_masks = [
        tf.io.decode_png(tf.io.read_file(f'{FLAGS.data_path}/{FLAGS.split}/masks/{scene_index}/{frame_index}.png'))[..., :3].numpy()
        for frame_index in frame_indices
    ]
    return convert_masks(np.stack(rgb_masks))


def load_clevr_masks(scene_index, frame_indices):

    def load_for_frame(frame_index):
        masks_for_frame = []
        object_index = 1
        while True:
            mask_filename = f'{FLAGS.data_path}/{FLAGS.split}/masks/{scene_index}/{frame_index}/{object_index}_modal001.png'
            if not os.path.exists(mask_filename):
                break
            masks_for_frame.append(tf.io.decode_png(tf.io.read_file(mask_filename))[..., 0] / 255)
            object_index += 1
        return tf.stack(masks_for_frame, axis=0)  # object, y, x

    fg_masks_by_frame = [load_for_frame(frame_index) for frame_index in frame_indices]
    assert all(len(masks_for_frame) == len(fg_masks_by_frame[0]) for masks_for_frame in fg_masks_by_frame)  # require same number of objects in every frame
    fg_masks_by_frame = tf.stack(fg_masks_by_frame, axis=0)  # frame, object, y, x

    bg_mask = 1 - tf.reduce_max(fg_masks_by_frame, axis=1, keepdims=True)
    masks_by_frame = tf.concat([bg_mask, fg_masks_by_frame], axis=1)  # put the background first, as the segmentation evaluation requires this!

    return torch.from_numpy(masks_by_frame.numpy()).permute(0, 2, 3, 1)  # frame, y, x, object


def get_background_slot_index(model, num_frames, num_scenes=100):
    
    # Run the model on single frames from many scenes, and check which slot is the most-common; assign this as background
    frames = [
        tf.io.decode_png(tf.io.read_file(f'{FLAGS.data_path}/{FLAGS.split}/images/{scene_index}/{scene_index % num_frames}.png'))[..., :3]
        for scene_index in range(num_scenes)
    ]
    frames = tf.cast(frames, tf.float32) / 255.
    _, _, soft_masks, _ = get_prediction(model, frames)  # frame, slot, y, x, singleton
    soft_masks = torch.from_numpy(soft_masks.numpy()).squeeze(-1)
    binary_masks = F.one_hot(soft_masks.argmax(dim=1), soft_masks.shape[1])  # frame, y, x, slot
    return binary_masks.sum(dim=(0, 1, 2)).argmax()


def main(argv):

    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    num_scenes = len(os.listdir(f'{FLAGS.data_path}/{FLAGS.split}/images'))
    num_frames = len(os.listdir(f'{FLAGS.data_path}/{FLAGS.split}/images/0'))  # we assume it's equal for all scenes

    assert FLAGS.num_scenes <= num_scenes
    print(f'evaluating on {FLAGS.num_scenes} / {num_scenes} scenes')
    num_scenes = FLAGS.num_scenes

    model = load_model(FLAGS.ckpt_path, num_slots=FLAGS.num_slots, num_iters=FLAGS.num_iterations, batch_size=1)

    background_slot_index = get_background_slot_index(model, num_frames)
    print(f'chose slot {background_slot_index} as background')

    metric_to_values = defaultdict(lambda: [])
    for scene_index in tqdm(range(num_scenes)):

        frame_indices = [scene_index % num_frames]  # match o3d-nerf evaluation, which uses one cyclically-varying frame per scene

        frames = [
            tf.io.decode_png(tf.io.read_file(f'{FLAGS.data_path}/{FLAGS.split}/images/{scene_index}/{frame_index}.png'))[..., :3]
            for frame_index in frame_indices
        ]
        frames = tf.cast(frames, tf.float32) / 255.
        assert frames.shape[1:3] == (FLAGS.resolution, FLAGS.resolution)

        gt_masks = (load_gqn_masks if 'gqn' in FLAGS.data_path else load_clevr_masks)(scene_index, frame_indices)  # frame, slot, y, x

        recon_combined, recons, masks, slots = get_prediction(model, frames)
        # recon_combined :: frame, y, x, rgb
        # recons :: frame, slot, y, x, rgb
        # masks :: frame, slot, y, x, singleton
        # slots :: frame, slot, channel

        # Calculate per-frame 2D metrics, only on the given frames
        for frame_index_index in range(len(frame_indices)):

            gt_frame = torch.from_numpy(frames[frame_index_index].numpy())
            recon_frame = torch.from_numpy(recon_combined[frame_index_index].numpy())
            soft_masks = torch.from_numpy(masks[frame_index_index].numpy()).squeeze(-1)  # slot, y, x
            gt_masks_for_frame = gt_masks[frame_index_index]

            mse = img2mse(gt_frame, recon_frame)
            metric_to_values['mse'].append(mse.item())
            metric_to_values['psnr'].append(mse2psnr(mse).item())

            binary_masks = F.one_hot(soft_masks.argmax(dim=0), num_classes=soft_masks.shape[0])  # y, x, slot
            binary_masks = torch.cat([
                binary_masks[..., :background_slot_index],
                binary_masks[..., background_slot_index + 1 :],
                binary_masks[..., background_slot_index : background_slot_index + 1]
            ], dim=-1)  # i.e. move background to last slot

            metric_to_values['segmentations_ARI_per_frame'].append(metrics.segmentation_ari(binary_masks, gt_masks_for_frame, foreground_only=False))
            metric_to_values['segmentations_fg_ARI_per_frame'].append(metrics.segmentation_ari(binary_masks, gt_masks_for_frame, foreground_only=True))
            metric_to_values['segmentations_FG_BG_ARI_per_frame'].append(metrics.segmentation_ari(binary_masks, gt_masks_for_frame, foreground_only=False, fg_bg_ari=True))
            metric_to_values['segmentations_IOU'].append(metrics.iou_for_frame(binary_masks, gt_masks_for_frame))
            metric_to_values['segmentations_FG_IOU'].append(metrics.iou_for_frame(binary_masks[..., :-1], gt_masks_for_frame[..., 1:]))

        if FLAGS.out_path != '':
            # Write reconstruction and mask images
            scene_out_path = f'{FLAGS.out_path}/{scene_index}'
            os.makedirs(scene_out_path, exist_ok=True)
            def write_png(rgb, suffix):
                tf.io.write_file(
                    f'{scene_out_path}/{frame_indices[frame_index_index]}_{suffix}.png',
                    tf.io.encode_png(tf.cast(tf.clip_by_value(rgb, 0., 1.) * 255., tf.uint8))
                )
            for frame_index_index in range(len(frame_indices)):
                write_png(recon_combined[frame_index_index], 'recon')
                for slot_index in range(masks.shape[1]):
                    write_png(tf.tile(masks[frame_index_index, slot_index], [1, 1, 3]), f'mask-{slot_index}')

        if False:
            # Visualize masked per-slot reconstructions
            import matplotlib.pyplot as plt
            image = frames[0]
            recon_combined = recon_combined[0]
            recons = recons[0]
            masks = masks[0]
            fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
            ax[0].imshow(image)
            ax[0].set_title('Image')
            ax[1].imshow(recon_combined)
            ax[1].set_title('Recon.')
            for i in range(num_slots):
                ax[i + 2].imshow(recons[i] * masks[i] + (1 - masks[i]))
                ax[i + 2].set_title('Slot %s' % str(i + 1))
            for i in range(len(ax)):
                ax[i].grid(False)
                ax[i].axis('off')
            plt.savefig('/root/workspace/wibl.png')

    with open(FLAGS.metrics_filename, 'wt') as f:
        for metric, values in metric_to_values.items():
            print(f'{metric}: {np.mean(values)}')
            f.write(f'{metric}: {np.mean(values)}\n')


if __name__ == '__main__':
    app.run(main)

