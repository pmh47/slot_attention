# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run (in container, on idagpu) with:
# PYTHONPATH=. python slot_attention/object_discovery/collect_slots.py --dataset ood --data_path /workspace/o3d-nerf/data/gqn --resolution 80 --num_slots 5 --ckpt_path /workspace/o3d-nerf/slot_attention/checkpoints/gqn_5-slot_seed-1_cvr-1e-2
# PYTHONPATH=. python slot_attention/object_discovery/collect_slots.py --dataset ood --data_path /workspace/o3d-nerf/data/arrow --resolution 96 --max_num_frames 10 --num_slots 5 --ckpt_path /workspace/o3d-nerf/slot_attention/checkpoints/arrow_seed-1

"""Training loop for object discovery with Slot Attention."""
import numpy as np
from tqdm import  tqdm

from absl import app
from absl import flags
import tensorflow as tf

import slot_attention.data as data_utils
import slot_attention.model as model_utils


FLAGS = flags.FLAGS
flags.DEFINE_string("ckpt_path", "/workspace/o3d-nerf/slot_attention/checkpoints/arrow_2022-04-19", "Where to load the checkpoint from.")
flags.DEFINE_string("dataset", "clevr", "Which dataset to use")
flags.DEFINE_string("data_path", "", "Root folder for the dataset (ignored for clevr)")
flags.DEFINE_integer("max_num_frames", 0, "Maximum number of frames per scene (ignored for clevr)")
flags.DEFINE_integer("resolution", 96, "Image resolution")
flags.DEFINE_integer("num_slots", 5, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Batch size for the model.")
flags.DEFINE_integer("num_batches", 100, "Number of batches of images to learn from.")


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
    assert False

  return model


def main(argv):
  del argv
  tf.random.set_seed(FLAGS.seed)
  resolution = (FLAGS.resolution, FLAGS.resolution)

  # Build dataset iterators, optimizers and model.
  if FLAGS.dataset == 'clevr':
    dataset_builder = data_utils.build_clevr
    crop = True
  elif FLAGS.dataset == 'ood':
    if 'arrow' in FLAGS.data_path and (FLAGS.max_num_frames != 10):
      assert False, 'for arrow, beware train images 10-30 have ood viewpoints! you probably want to set --max_num_frames 10'
    dataset_builder = lambda **kwargs: data_utils.build_ood(data_path=FLAGS.data_path, max_num_frames=FLAGS.max_num_frames,**kwargs)
    crop = False
  else:
    raise RuntimeError('unknown dataset')

  data_iterator = data_utils.build_iterator(
      dataset_builder, FLAGS.batch_size, split="train", resolution=resolution, shuffle=True,
      max_n_objects=6, get_properties=False, apply_crop=crop)

  model = load_model(FLAGS.ckpt_path, num_slots=FLAGS.num_slots, num_iters=FLAGS.num_iterations, batch_size=FLAGS.batch_size)

  all_slots = []
  for _ in tqdm(range(FLAGS.num_batches)):

    batch = next(data_iterator)

    recon_combined, recons, masks, slots = model(batch['image'])
    # `slots` has shape: [batch_size, num_slots, slot_size].

    all_slots.append(slots)

  all_slots = np.concatenate(all_slots, axis=0)
  np.save(FLAGS.ckpt_path + '/slots.npy', all_slots)


if __name__ == "__main__":
  app.run(main)
