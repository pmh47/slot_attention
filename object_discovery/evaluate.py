import os
import numpy as np
import sklearn.mixture
from absl import app
from absl import flags
from tqdm import tqdm
from collections import defaultdict

import tensorflow as tf
import tensorflow_probability as tfp
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
flags.DEFINE_string("metrics_filename", "/tmp/statistics", "Filename to write final metric values as text.")
flags.DEFINE_integer("resolution", 96, "Image resolution")
flags.DEFINE_integer("num_slots", 5, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_integer("num_scenes", 100, "Number of scenes to evaluate.")
flags.DEFINE_boolean("mcmc", False, "Use MCMC inference.")
flags.DEFINE_integer("num_cpts", 40, "Number of GMM components.")

SLOT_SIZE = 64  # this is hard-coded in the original SlotAttention class!


def load_model(checkpoint_dir, num_slots=11, num_iters=3, batch_size=16):
    model = model_utils.build_model(
        (FLAGS.resolution, FLAGS.resolution), batch_size, num_slots, num_iters,
        model_type="object_discovery" if not FLAGS.mcmc else "object_decoder")

    ckpt = tf.train.Checkpoint(network=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=checkpoint_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Restored from", ckpt_manager.latest_checkpoint)
    else:
        print("Initialised model from scratch")

    return model


def renormalize(x):
    """Renormalize from [-1, 1] to [0, 1]."""
    return x / 2. + 0.5


def get_prediction(model, image):

    recon_combined, recons, masks, slots = model(image * 2. - 1.)
    recon_combined = renormalize(recon_combined)
    recons = renormalize(recons)

    return recon_combined, recons, masks, slots


class MCMC:

    def __init__(self):

        slots = np.load(FLAGS.ckpt_path + '/slots.npy')
        slots = slots.reshape(-1, slots.shape[-1])
        np.random.shuffle(slots)
        slots = slots[:1000]
        print(slots.shape)
        self._gmm = sklearn.mixture.GaussianMixture(FLAGS.num_cpts, covariance_type='diag', verbose=True)
        self._gmm.fit(slots)

    def __call__(self, model, frames):

        assert frames.shape[0] == 1

        iterations = 5000
        mh_frequency = 10
        mala_adam_lr = 1.e-3
        epsilon = 1.e-4  # std of LD normal perturbation
        sigma = 0.025  # std of pixel likelihod

        current_slots = tf.Variable(self._gmm.sample(FLAGS.num_slots)[0], dtype=tf.float32)  # slot, channel
        current_mala_weights = None

        def get_mse(slots):
            recon_combined, _, _, _ = model(slots[None])
            recon_combined = renormalize(recon_combined)  # 1, y, x, rgb
            return tf.reduce_mean(tf.square(recon_combined - frames))

        def get_gmm_proposal():

            slot_index = tf.random.uniform(shape=[], maxval=current_slots.shape[0], dtype=tf.int32)
            slot_current_value = current_slots[slot_index]
            slot_new_value = self._gmm.sample()[0][0]
            proposal_slots = tf.tensor_scatter_nd_update(current_slots, [[slot_index]], [slot_new_value])
            log_Q_current_given_candidate = self._gmm.score([slot_current_value])
            log_Q_candidate_given_current = self._gmm.score([slot_new_value])

            return proposal_slots, None, log_Q_current_given_candidate, log_Q_candidate_given_current

        def get_ld_proposal():

            slots_copy = tf.Variable(current_slots)
            optimizer = tf.keras.optimizers.Adam(learning_rate=mala_adam_lr)

            def step():
                with tf.GradientTape() as tape:
                    mse_loss = get_mse(slots_copy)
                d_loss_d_slots = tape.gradient(mse_loss, slots_copy)
                optimizer.apply_gradients([(d_loss_d_slots, slots_copy)])

            if current_mala_weights is not None:
                step()  # force the optimizer to create its slots, so set_weights works
                slots_copy.assign(current_slots)
                optimizer.set_weights(current_mala_weights)

            step()
            proposal_mala_weights = optimizer.get_weights()
            dist_from_current = tfp.distributions.Normal(slots_copy, epsilon)
            proposal_slots = dist_from_current.sample()
            slots_copy.assign(proposal_slots)
            step()
            dist_from_proposal = tfp.distributions.Normal(slots_copy, epsilon)

            log_Q_candidate_given_current = tf.reduce_sum(dist_from_current.log_prob(proposal_slots))
            log_Q_current_given_candidate = tf.reduce_sum(dist_from_proposal.log_prob(current_slots))

            return proposal_slots, proposal_mala_weights, log_Q_current_given_candidate, log_Q_candidate_given_current

        for iteration in range(iterations):

            if iteration % mh_frequency == 0:
                proposal_slots, proposal_mala_weights, log_Q_current_given_candidate, log_Q_candidate_given_current = get_gmm_proposal()
            else:
                proposal_slots, proposal_mala_weights, log_Q_current_given_candidate, log_Q_candidate_given_current = get_ld_proposal()

            log_P_current = -get_mse(current_slots) / sigma**2  # ** we could cache this!
            log_P_candidate = -get_mse(proposal_slots) / sigma**2  # ** we could also cache this for LD proposals
            # log_alpha = (log_P_candidate - log_Q_candidate_given_current) - (log_P_current - log_Q_current_given_candidate)
            log_alpha = log_P_candidate - log_P_current
            criterion = min(1, np.exp(log_alpha))

            # print(criterion, log_P_current, log_P_candidate, log_Q_current_given_candidate, log_Q_candidate_given_current)

            if tf.random.uniform(shape=[]) < criterion:  # i.e. accept the proposal
            # if log_P_candidate > log_P_current:
            #     print('accept')
                current_slots.assign(proposal_slots)
                if proposal_mala_weights is not None:
                    current_mala_weights = proposal_mala_weights
            else:
                # print(f'reject ({criterion:.2f})')
                pass

            if iteration % 100 == 0:
                print(f'{iteration}: mse = {log_P_current}')
                tf.io.write_file(f'{iteration:04}.png', tf.image.encode_png(tf.cast(tf.concat([
                    tf.clip_by_value(renormalize(model(current_slots[None])[0][0]), 0., 1.),
                    frames[0]
                ], axis=1) * 255, tf.uint8)))

        recon_combined, recons, masks, slots = model(current_slots[None])
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


def masks_to_rgb_segmentation(masks):

    # masks :: y, x, slot; assumed to be 1-hit, with last slot being background

    segmentation_colours = np.asarray(
        [[50, 168, 82], [50, 82, 168], [153, 140, 38], [158, 30, 105], [158, 30, 30], [87, 156, 79],
         [50, 20, 82], [50, 10, 168], [10, 140, 38], [250, 30, 105], [158, 250, 250], [250, 20, 79]],
    ) / 255.

    return (masks[..., :-1, None] * segmentation_colours[:masks.shape[2] - 1]).sum(axis=-2)


def main(argv):

    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    num_scenes = len(os.listdir(f'{FLAGS.data_path}/{FLAGS.split}/images'))
    num_frames = len(os.listdir(f'{FLAGS.data_path}/{FLAGS.split}/images/0'))  # we assume it's equal for all scenes

    assert FLAGS.num_scenes <= num_scenes
    print(f'evaluating on {FLAGS.num_scenes} / {num_scenes} scenes')
    num_scenes = FLAGS.num_scenes

    model = load_model(FLAGS.ckpt_path, num_slots=FLAGS.num_slots, num_iters=FLAGS.num_iterations, batch_size=1)

    predictor = MCMC() if FLAGS.mcmc else get_prediction

    metric_to_values = defaultdict(lambda: [])
    for scene_index in tqdm(range(num_scenes)):

        frame_indices = [scene_index % num_frames]  # match o3d-nerf evaluation, which uses one cyclically-varying frame per scene

        frames = [
            tf.io.decode_png(tf.io.read_file(f'{FLAGS.data_path}/{FLAGS.split}/images/{scene_index}/{frame_index}.png'))[..., :3]
            for frame_index in frame_indices
        ]
        frames = tf.cast(frames, tf.float32) / 255.
        assert frames.shape[1:3] == (FLAGS.resolution, FLAGS.resolution)

        gt_masks = (load_gqn_masks if 'gqn' in FLAGS.data_path else load_clevr_masks)(scene_index, frame_indices)  # frame, y, x, slot

        recon_combined, recons, masks, slots = predictor(model, frames)
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
            # background_slot_index = binary_masks.sum(dim=(0, 1)).argmax()  # decide this per-frame, as the random slot initialisation may result in inconsistent assignments
            gt_bg_mask = gt_masks_for_frame[..., :1]  # y, x, singleton
            background_slot_index = ((binary_masks * gt_bg_mask).sum((0, 1)) / (torch.maximum(binary_masks, gt_bg_mask).sum((0, 1)) + 1.)).argmax()  # decide this per-frame, as the random slot initialisation may result in inconsistent assignments
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
                write_png(frames[frame_index_index], 'input')
                write_png(recon_combined[frame_index_index], 'recon')
                for slot_index in range(masks.shape[1]):
                    write_png(tf.tile(masks[frame_index_index, slot_index], [1, 1, 3]), f'mask-{slot_index}')
                    masked_rgb = recons[frame_index_index, slot_index] * masks[frame_index_index, slot_index] + (1. - masks[frame_index_index, slot_index])
                    write_png(masked_rgb, f'masked-rgb-{slot_index}')
                write_png(masks_to_rgb_segmentation(binary_masks), 'segmentation')

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

    with open(FLAGS.metrics_filename + '.txt', 'wt') as f:
        for metric, values in metric_to_values.items():
            print(f'{metric}: {np.nanmean(values)}')
            f.write(f'{metric}: {np.nanmean(values)}\n')
    with open(FLAGS.metrics_filename + '.metrics.txt', 'wt') as f:
        for metric, values in metric_to_values.items():
            f.write(f'{metric},')
        f.write('\n')
    with open(FLAGS.metrics_filename + '.mean.txt', 'wt') as f:
        for metric, values in metric_to_values.items():
            f.write(f'{np.nanmean(values)},')
        f.write('\n')
    with open(FLAGS.metrics_filename + '.std.txt', 'wt') as f:
        for metric, values in metric_to_values.items():
            f.write(f'{np.nanstd(values)},')
        f.write('\n')


if __name__ == '__main__':
    app.run(main)

