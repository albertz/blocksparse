

import numpy
import networkx  # for the pattern
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell


class Linear:
  def __init__(self, seed, block_size=32, connectivity=5, mul_feature_axis=0, feature_axis=-1):
    """
    :param int seed: for the random sparsity pattern(s)
    :param int block_size: for BlocksparseMatMul
    :param int connectivity: used for :func:`sparsity_pattern_barabasi_albert`
    :param int mul_feature_axis: for BlocksparseMatMul
    :param int feature_axis: specifies the feature axis of the in/out values, see :func:`self.__call__`
    """
    self.block_size = block_size
    self.connectivity = connectivity
    self.mul_feature_axis = mul_feature_axis
    self.feature_axis = feature_axis
    self.random = numpy.random.RandomState(seed)

  @staticmethod
  def move_feature_axis(x, old_axis, new_axis):
    """
    :param tf.Tensor x:
    :param int old_axis:
    :param int new_axis:
    :rtype: tf.Tensor
    """
    ndim = x.get_shape().ndims
    assert ndim is not None, "not supported currently: %r" % x
    if old_axis < 0:
      old_axis += ndim
      assert old_axis >= 0
    if new_axis < 0:
      new_axis += ndim
      assert new_axis >= 0
    if old_axis == new_axis:
      return x
    perm = list(range(ndim))
    old = perm.pop(old_axis)
    perm.insert(new_axis, old)
    return tf.transpose(x, perm, name="move_feature_axis")

  def __call__(self, x, output_dim, feature_axis=None):
    """
    :param tf.Tensor x: (..., input_dim) (if feature_axis = -1)
    :param int output_dim:
    :param int feature_axis: specifies the feature axis of `x` and the return value
    :return: x.shape[:-1] + [output_dim] (if feature_axis = -1)
    :rtype: tf.Tensor
    """
    block_size = self.block_size
    seed = self.random.randint(2 ** 31)
    if feature_axis is None:
      feature_axis = self.feature_axis
    x_dims = x.get_shape().dims
    input_dim = x_dims[feature_axis].value
    assert input_dim is not None, "%r shape unknown" % (x,)
    assert input_dim % block_size == 0 and output_dim % block_size == 0

    from blocksparse.matmul import BlocksparseMatMul
    sparsity = sparsity_pattern_barabasi_albert(
      n1=input_dim // block_size, n2=output_dim // block_size, m=self.connectivity, seed=seed)
    bsmm = BlocksparseMatMul(sparsity, block_size=block_size, feature_axis=self.mul_feature_axis)
    weights = tf.get_variable("W", shape=bsmm.w_shape)

    x = self.move_feature_axis(x, old_axis=feature_axis, new_axis=self.mul_feature_axis)
    y = bsmm(x, weights)
    assert isinstance(x, tf.Tensor)
    y = self.move_feature_axis(y, old_axis=self.mul_feature_axis, new_axis=feature_axis)
    y_dims = list(x_dims)
    y_dims[feature_axis] = output_dim
    y.set_shape(y_dims)
    return y


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


class BlocksparseLSTMCell(rnn_cell.RNNCell):
  """
  Standard LSTM but uses OpenAI blocksparse kernels to support bigger matrices.

  Refs:

    https://blog.openai.com/block-sparse-gpu-kernels/
    https://github.com/openai/blocksparse
    https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf

  """

  def __init__(self, num_units, seed, block_size=32, connectivity=5):
    """
    :param int num_units:
    :param int seed:
    :param int block_size:
    :param int connectivity:
    """
    super(BlocksparseLSTMCell, self).__init__()
    self.num_units = num_units
    self.linear = Linear(block_size=block_size, connectivity=connectivity, seed=seed)

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
      x += tf.get_variable("b", shape=(dim,))
      x.set_shape((None, None, dim))  # (time,batch,dim)
      return x

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


class BlocksparseMultiplicativeMultistepLSTMCell(rnn_cell.RNNCell):
  """
  Multiplicative LSTM with multiple steps, as in the OpenAI blocksparse paper.
  Uses OpenAI blocksparse kernels to support bigger matrices.

  Refs:

    https://blog.openai.com/block-sparse-gpu-kernels/
    https://github.com/openai/blocksparse
    https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf

  """

  _num_parts_step1 = 3  # additive, multiplicative
  _num_parts_step_final = 4  # cell-in, gate-in, gate-forget, gate-out

  def __init__(self, num_units, depth, seed, block_size=32, connectivity=5):
    """
    :param int num_units:
    :param int depth: internal depth
    :param int seed:
    :param int block_size:
    :param int connectivity:
    """
    assert depth >= 1
    super(BlocksparseMultiplicativeMultistepLSTMCell, self).__init__()
    self.num_units = num_units
    self.depth = depth
    self.linear = Linear(block_size=block_size, connectivity=connectivity, seed=seed)

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
      dim = self.num_units * self._num_parts_step1
      x = self.linear(x, dim)
      x += tf.get_variable("b", shape=(dim,))
      x.set_shape((None, None, dim))  # (time,batch,dim)
      return x

  def call(self, inputs, state):
    """
    :param tf.Tensor inputs: (batch, num_units * 4)
    :param rnn_cell.LSTMStateTuple state: (batch, num_units)
    :return: output, new_state
    """
    assert isinstance(inputs, tf.Tensor)
    assert isinstance(state, rnn_cell.LSTMStateTuple)
    # All internal steps performed with moved feature axis, should be faster.
    x = self.linear.move_feature_axis(inputs, old_axis=-1, new_axis=self.linear.mul_feature_axis)
    h = self.linear.move_feature_axis(state.h, old_axis=-1, new_axis=self.linear.mul_feature_axis)
    c = self.linear.move_feature_axis(state.c, old_axis=-1, new_axis=self.linear.mul_feature_axis)

    # This is counted as step 1.
    with tf.variable_scope("step0"):
      dim = self.num_units * self._num_parts_step1
      # Note: inputs has a bias already.
      x += self.linear(h, dim, feature_axis=self.linear.mul_feature_axis)
      x1, x2, x3 = tf.split(x, self._num_parts_step1, axis=self.linear.mul_feature_axis)
      x = tf.nn.relu(x1 + x2 * x3)

    for step in range(1, self.depth):
      with tf.variable_scope('step%i' % step):
        dim = self.num_units
        x = self.linear(x, dim, feature_axis=self.linear.mul_feature_axis)
        x += tf.expand_dims(tf.get_variable("b", shape=(dim,)), axis=self.linear.mul_feature_axis + 1)
        x = tf.nn.relu(x)
        x.set_shape((dim if self.linear.mul_feature_axis == 0 else None, None))

    with tf.variable_scope("gating"):
      dim = self.num_units * self._num_parts_step_final
      x = self.linear(x, dim, feature_axis=self.linear.mul_feature_axis)
      cell_in, gate_in, gate_forget, gate_out = tf.split(
        x, self._num_parts_step_final, axis=self.linear.mul_feature_axis)
      cell_in = tf.tanh(cell_in)
      gate_in = tf.sigmoid(gate_in)
      gate_forget = tf.sigmoid(gate_forget)
      gate_out = tf.sigmoid(gate_out)
      cell = c * gate_forget + cell_in * gate_in
      out = tf.tanh(cell) * gate_out

    # Move feature axis back to where it is expected.
    cell = self.linear.move_feature_axis(cell, old_axis=self.linear.mul_feature_axis, new_axis=-1)
    out = self.linear.move_feature_axis(out, old_axis=self.linear.mul_feature_axis, new_axis=-1)
    cell.set_shape((None, self.num_units))
    out.set_shape((None, self.num_units))
    assert out.get_shape().dims[0].value == cell.get_shape().dims[0].value == inputs.get_shape().dims[0].value, 'b.dim'
    return out, rnn_cell.LSTMStateTuple(c=cell, h=out)
