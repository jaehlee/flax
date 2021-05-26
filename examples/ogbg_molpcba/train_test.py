# Copyright 2021 The Flax Authors.
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

"""Tests for flax.examples.ogbg_molpcba.train."""

import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import flax
import input_pipeline
import train
from configs import default
from configs import test
from flax.training import train_state
import jax
from jax import numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def average_with_mask(arr: jnp.ndarray, mask: jnp.ndarray):
  """Returns the average over elements where mask is True."""
  arr = jnp.where(mask, arr, 0)
  return jnp.sum(arr) / jnp.sum(mask)


class OgbgMolpcbaTrainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Print the current platform (the default device).
    platform = jax.local_devices()[0].platform
    print('Running on platform:', platform.upper())

    # Create PRNG keys.
    self.rng = jax.random.PRNGKey(0)

    # Create dummy datasets.
    self.dataset_length = 20
    self.datasets = input_pipeline.get_dummy_datasets(
        self.dataset_length, batch_size=10)
    self.unbatched_datasets = input_pipeline.get_dummy_datasets(
        self.dataset_length, batch_size=None)

  @parameterized.product(
      probs=[[[0.8, 0.9, 0.3, 0.5]]],
      labels=[[[1, 0, 1, 1]], [[1, 0, 1, jnp.nan]], [[1, 0, jnp.nan, jnp.nan]],
              [[1, jnp.nan, jnp.nan, jnp.nan]]],
  )
  def test_binary_cross_entropy_loss(self, probs, labels):
    probs = jnp.asarray(probs)
    labels = jnp.asarray(labels)

    logits = jnp.log(probs / (1 - probs))
    mask = ~jnp.isnan(labels)

    loss_array = train.binary_cross_entropy_with_mask(
        logits=logits, labels=labels, mask=mask)
    loss = average_with_mask(loss_array, mask)
    expected_loss_array = -(jnp.log(probs) * labels) - (
        jnp.log(1 - probs) * (1 - labels))
    expected_loss = average_with_mask(expected_loss_array, mask)

    self.assertAlmostEqual(loss, expected_loss, places=5)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_valid_tasks',
          logits=[[-1., 1.], [1., 1.], [2., -1.]],
          labels=[[jnp.nan, jnp.nan], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan]],
          expected_result=jnp.nan),
      dict(
          testcase_name='1_valid_task',
          logits=[[-1., 1.], [1., 1.], [2., -1.]],
          labels=[[0, jnp.nan], [1, jnp.nan], [1, jnp.nan]],
          expected_result=1.),
      dict(
          testcase_name='2_valid_tasks',
          logits=[[-1., 1.], [1., 1.], [2., -1.]],
          labels=[[0, jnp.nan], [1, 0], [1, 1]],
          expected_result=0.75),
  )
  def test_mean_average_precision(self, logits, labels, expected_result):
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    mask = ~jnp.isnan(labels)

    mean_average_precision = train.MeanAveragePrecision.from_model_output(
        logits=logits, labels=labels, mask=mask).compute()

    if jnp.isnan(expected_result):
      self.assertTrue(jnp.isnan(mean_average_precision))
    else:
      self.assertAlmostEqual(expected_result, mean_average_precision)

  @parameterized.parameters(
      dict(
          loss=[[0.5, 1.], [1.5, 1.3], [2., 1.2]],
          logits=[[-1., 1.], [1., 1.], [2., 0.]],
          labels=[[0, jnp.nan], [1, 0], [0, 1]],
          mask=[[True, False], [True, True], [False, False]],
          expected_results={'loss': 1.1, 'accuracy': 2/3,
                            'mean_average_precision': 1.0}),
  )
  def test_eval_metrics(self, loss, logits, labels, mask, expected_results):
    loss = jnp.asarray(loss)
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    mask = jnp.asarray(mask)

    eval_metrics = train.EvalMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=labels, mask=mask).compute()
    for metric in expected_results:
      self.assertAlmostEqual(expected_results[metric], eval_metrics[metric])

  @parameterized.parameters(
      dict(loss=[[0.5, 1.], [1.5, 1.3], [2., 1.2]],
           logits=[[-1., 1.], [1., 1.], [2., 0.]],
           labels=[[0, jnp.nan], [1, 0], [0, 1]],
           mask=[[True, False], [True, True], [False, False]],
           expected_results={'loss': 1.1, 'accuracy': 2/3}),
  )
  def test_train_metrics(self, loss, logits, labels, mask, expected_results):
    loss = jnp.asarray(loss)
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    mask = jnp.asarray(mask)

    train_metrics = train.TrainMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=labels, mask=mask).compute()
    for metric in expected_results:
      self.assertAlmostEqual(expected_results[metric], train_metrics[metric])

  def test_train_step(self):
    # Get the default configuration.
    config = default.get_config()

    # Initialize the network with a dummy graph.
    rng, init_rng = jax.random.split(self.rng)
    init_graphs = next(self.datasets['train'].as_numpy_iterator())
    init_graphs_preprocessed = train.zero_out_globals(init_graphs)
    init_net = train.create_model(config, deterministic=True)
    params = jax.jit(init_net.init)(init_rng, init_graphs_preprocessed)

    # Create the optimizer.
    optimizer = train.create_optimizer(config)

    # Create the training state.
    net = train.create_model(config, deterministic=False)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=optimizer)

    # Perform one step of updates.
    # We use the same batch of graphs that we used for initialization.
    state, train_metrics = train.train_step(
        state, init_graphs, rngs={'dropout': rng})

    # Check that none of the parameters are NaNs!
    params = flax.core.unfreeze(state.params)
    flat_params = {
        '/'.join(k): v
        for k, v in flax.traverse_util.flatten_dict(params).items()
    }
    for array in flat_params.values():
      self.assertTrue(jnp.all(~jnp.isnan(array)))

    # Check that the metrics are well defined.
    train_metrics_vals = train_metrics.compute()
    self.assertGreaterEqual(train_metrics_vals['loss'], 0)
    self.assertGreaterEqual(train_metrics_vals['accuracy'], 0)
    self.assertLessEqual(train_metrics_vals['accuracy'], 1)

  def test_evaluate_step(self):
    # Get the default configuration.
    config = default.get_config()

    # Initialize the network with a dummy graph.
    _, init_rng = jax.random.split(self.rng)
    init_graphs = next(self.datasets['train'].as_numpy_iterator())
    init_graphs_preprocessed = init_graphs._replace(
        globals=jnp.zeros([init_graphs.n_node.shape[0], 1]))
    init_net = train.create_model(config, deterministic=True)
    params = jax.jit(init_net.init)(init_rng, init_graphs_preprocessed)

    # Create the optimizer.
    optimizer = train.create_optimizer(config)

    # Create the evaluation state.
    eval_net = train.create_model(config, deterministic=True)
    eval_state = train_state.TrainState.create(
        apply_fn=eval_net.apply, params=params, tx=optimizer)

    # Perform one step of evaluation.
    # We use the same batch of graphs that we used for initialization.
    evaluate_metrics = train.evaluate_step(eval_state, init_graphs)

    # Check that the metrics are well defined.
    evaluate_metrics_vals = evaluate_metrics.compute()
    self.assertGreaterEqual(evaluate_metrics_vals['loss'], 0)
    self.assertGreaterEqual(evaluate_metrics_vals['accuracy'], 0)
    self.assertLessEqual(evaluate_metrics_vals['accuracy'], 1)
    self.assertGreaterEqual(evaluate_metrics_vals['mean_average_precision'], 0)
    self.assertLessEqual(evaluate_metrics_vals['mean_average_precision'], 1)

  def test_train_and_evaluate(self):
    # Create a temporary directory where TensorBoard metrics are written.
    workdir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    # Get the test configuration.
    config = test.get_config()

    # Ensure train_and_evaluate() runs without any errors!
    def as_dataset_fn(*args, **kwargs):
      del args, kwargs
      return self.unbatched_datasets['train']

    with tfds.testing.mock_data(as_dataset_fn=as_dataset_fn):
      train.train_and_evaluate(config=config, workdir=workdir)


if __name__ == '__main__':
  absltest.main()
