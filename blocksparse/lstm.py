

import numpy
import networkx  # for the pattern
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell


class BlocksparseLSTMCell(rnn_cell.RNNCell):
  """
  Standard LSTM but uses OpenAI blocksparse kernels to support bigger matrices.

  Refs:

    https://blog.openai.com/block-sparse-gpu-kernels/
    https://github.com/openai/blocksparse
    https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf

  It uses our own wrapper, see :func:`TFNativeOp.init_blocksparse`.
  """

  def __init__(self, num_units, block_size=32, connectivity=5, seed=None):
    """
    :param int num_units:
    """
    super(BlocksparseLSTMCell, self).__init__()
    self.num_units = num_units
    self.block_size = block_size
    self.connectivity = connectivity
    self.random = numpy.random.RandomState(seed)

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(c=self.num_units, h=self.num_units)

  def get_input_transformed(self, x):
    """
    :param tf.Tensor x:
    :rtype: tf.Tensor
    """
    with tf.variable_scope('input'):
      dim = self.num_units * 4
      x = self.linear(x, dim)
      bias = tf.get_variable("b", shape=(dim,))
      return x + bias

  def linear(self, x, output_dim, block_size=None, connectivity=None, seed=None):
    """
    :param tf.Tensor x: (..., input_dim)
    :param int output_dim:
    :param int|None block_size:
    :param int|None connectivity:
    :param int|None seed:
    :return: x.shape[:-1] + [output_dim]
    :rtype: tf.Tensor
    """
    if block_size is None:
      block_size = self.block_size
    if connectivity is None:
      connectivity = self.connectivity
    if seed is None:
      seed = self.random.randint(2 ** 31)
    x_dims = x.get_shape().dims
    input_dim = x_dims[-1].value
    assert input_dim is not None, "%r shape unknown" % (x,)
    assert input_dim % block_size == 0 and output_dim % block_size == 0

    from blocksparse.matmul import BlocksparseMatMul
    sparsity = sparsity_pattern_barabasi_albert(
      n1=input_dim // block_size, n2=output_dim // block_size, m=connectivity, seed=seed)
    bsmm = BlocksparseMatMul(sparsity, block_size=block_size, feature_axis=0)
    weights = tf.get_variable("W", shape=bsmm.w_shape)
    x = tf.transpose(x, [len(x_dims) - 1] + list(range(len(x_dims) - 1)))  # move feature axis
    y = bsmm(x, weights)
    assert isinstance(x, tf.Tensor)
    y = tf.transpose(y, list(range(1, len(x_dims))) + [0])  # move feature axis
    y.set_shape(x_dims[:-1] + [output_dim])
    return y

  def call(self, inputs, state):
    assert isinstance(inputs, tf.Tensor)
    assert isinstance(state, rnn_cell.LSTMStateTuple)

    dim = self.num_units * 4
    x = self.linear(state.h, dim) + inputs
    cell_in, gate_in, gate_forget, gate_out = tf.split(x, 4, axis=-1)
    cell_in = tf.tanh(cell_in)
    gate_in = tf.sigmoid(gate_in)
    gate_forget = tf.sigmoid(gate_forget)
    gate_out = tf.sigmoid(gate_out)
    cell = state.c * gate_forget + cell_in * gate_in
    out = tf.tanh(cell) * gate_out
    return out, rnn_cell.LSTMStateTuple(c=cell, h=out)


def sparsity_pattern_square_barabasi_albert(n, m, seed):
  """
  :param int n:
  :param int m: 1 <= m <= n
  :param int seed:
  :return: matrix (n,n), int32
  :rtype: numpy.ndarray
  """
  g = networkx.generators.barabasi_albert_graph(n=n, m=m, seed=seed)
  a = networkx.adjacency_matrix(g).toarray().astype(numpy.int32) + numpy.eye(n, dtype=numpy.int32)
  a[0:m, 0:m] = 1
  return a


def sparsity_pattern_barabasi_albert(n1, n2, m, seed):
  """
  :param int n1: multiple of n2
  :param int n2: multiple of n1
  :param int m: 1 <= m <= min(n1, n2)
  :param int seed:
  :return: matrix (n1,n2)
  :rtype: numpy.ndarray
  """
  if n1 == n2:
    return sparsity_pattern_square_barabasi_albert(n=n1, m=m, seed=seed)
  if n1 > n2:
    return sparsity_pattern_barabasi_albert(n1=n2, n2=n1, m=m, seed=seed).transpose()
  assert n2 >= n1 and n2 % n1 == 0
  assert m <= n1
  random = numpy.random.RandomState(seed)
  seeds = [random.randint(2 ** 31) for i in range(n2 // n1)]
  parts = [
    sparsity_pattern_square_barabasi_albert(n=n1, m=m, seed=seed)
    for seed in seeds]
  a = numpy.concatenate(parts, axis=1)
  assert a.shape == (n1, n2)
  return a
