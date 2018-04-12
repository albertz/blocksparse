

import numpy
import networkx  # for the pattern
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell


class BlocksparseLinear:
  def __init__(self, seed, block_size=32, connectivity=5, mul_feature_axis=0, feature_axis=-1,
               layer_norm=True, fast_layer_norm=True,
               always_dense=False):
    """
    :param int seed: for the random sparsity pattern(s)
    :param int block_size: for BlocksparseMatMul
    :param int connectivity: used for :func:`sparsity_pattern_barabasi_albert`
    :param int mul_feature_axis: for BlocksparseMatMul
    :param int feature_axis: specifies the feature axis of the in/out values, see :func:`self.__call__`
    :param bool layer_norm: apply layer normalization in each linear call
    :param bool fast_layer_norm: layer norm implementation by OpenAI
    :param bool always_dense:
    """
    self.block_size = block_size
    self.connectivity = connectivity
    self.mul_feature_axis = mul_feature_axis
    self.feature_axis = feature_axis
    self.layer_norm = layer_norm
    self.fast_layer_norm = fast_layer_norm
    self.always_dense = always_dense
    self.random = numpy.random.RandomState(seed)
    self.matmuls = []

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

  def __call__(self, x, output_dim, feature_axis=None, with_bias=True, dense=False):
    """
    :param tf.Tensor x: (..., input_dim) (if feature_axis = -1)
    :param int output_dim:
    :param int feature_axis: specifies the feature axis of `x` and the return value
    :param bool with_bias:
    :param bool dense:
    :return: x.shape[:-1] + [output_dim] (if feature_axis = -1)
    :rtype: tf.Tensor
    """
    block_size = self.block_size
    seed = self.random.randint(2 ** 31)
    if feature_axis is None:
      feature_axis = self.feature_axis
    mul_feature_axis = self.mul_feature_axis
    x_dims = x.get_shape().dims
    input_dim = x_dims[feature_axis].value
    assert input_dim is not None, "%r shape unknown" % (x,)

    if self.always_dense:
      dense = True
    if dense:
      mul_feature_axis = -1
    if mul_feature_axis < 0:
      mul_feature_axis += len(x_dims)
      assert mul_feature_axis >= 0
    x = self.move_feature_axis(x, old_axis=feature_axis, new_axis=mul_feature_axis)

    if dense:
      bsmm = None
      weights = tf.get_variable("W", shape=(input_dim, output_dim))
      if len(x_dims) > 2:
        x_shape = tf.shape(x)
        x = tf.reshape(x, (-1, input_dim), name='reshape_x')
      else:
        x_shape = None
      y = tf.matmul(x, weights)
      if len(x_dims) > 2:
        y = tf.reshape(y, [x_shape[i] for i in range(len(x_dims) - 1)] + [output_dim], name="reshape_y")
    else:
      assert input_dim % block_size == 0 and output_dim % block_size == 0
      from blocksparse.matmul import BlocksparseMatMul
      sparsity_pattern = sparsity_pattern_barabasi_albert(
        n1=input_dim // block_size, n2=output_dim // block_size, m=self.connectivity, seed=seed)
      bsmm = BlocksparseMatMul(sparsity_pattern, block_size=block_size, feature_axis=mul_feature_axis)
      weights = tf.get_variable("W", shape=bsmm.w_shape)
      y = bsmm(x, weights)
    assert isinstance(y, tf.Tensor)

    if self.layer_norm and self.fast_layer_norm and mul_feature_axis == len(x_dims) - 1:
      # OpenAI kernel seems broken with axis != -1.
      # See RETURNN test case test_layer_norms.
      bias = tf.get_variable("b", shape=(output_dim,), initializer=tf.zeros_initializer())
      gain = tf.get_variable("g", shape=(output_dim,), initializer=tf.ones_initializer())
      from blocksparse.norms import layer_norm
      y = layer_norm(y, g=gain, b=bias, axis=mul_feature_axis)

    else:
      if self.layer_norm:
        # see :func:`tensorflow.contrib.layers.layer_norm`
        # Epsilon: OpenAI uses 1e-6, TF contrib uses 1e-12, pbhatia243 uses 1e-5.
        epsilon = 1e-6
        m, v = tf.nn.moments(y, axes=[mul_feature_axis], keep_dims=True)
        inv = tf.rsqrt(v + epsilon)
        gain = tf.get_variable("g", shape=(output_dim,), initializer=tf.ones_initializer())
        gain_bc = tf.reshape(gain, [output_dim if i == mul_feature_axis else 1 for i in range(len(x_dims))], name="g_bc")
        inv *= gain_bc
        y = y * inv - m * inv

      if with_bias:
        bias = tf.get_variable("b", shape=(output_dim,), initializer=tf.zeros_initializer())
        bias_bc = tf.reshape(bias, [output_dim if i == mul_feature_axis else 1 for i in range(len(x_dims))], name="b_bc")
        y += bias_bc
      else:
        bias = None

    y = self.move_feature_axis(y, old_axis=mul_feature_axis, new_axis=feature_axis)
    y_dims = list(x_dims)
    y_dims[feature_axis] = output_dim
    y.set_shape(y_dims)
    self.matmuls.append({"bsmm": bsmm, "x": x, "y": y, "weights": weights, "bias": bias})
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
    self.linear = BlocksparseLinear(block_size=block_size, connectivity=connectivity, seed=seed)

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
      x += tf.get_variable("b", shape=(dim,), initializer=tf.zeros_initializer())
      x.set_shape((None, None, dim))  # (time,batch,dim)
      return x

  # noinspection PyMethodOverriding
  def call(self, inputs, state):
    assert isinstance(inputs, tf.Tensor)
    assert isinstance(state, rnn_cell.LSTMStateTuple)

    dim = self.num_units * 4
    x = self.linear(state.h, dim, with_bias=False) + inputs
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

  def __init__(self, num_units, depth, dense_input_transform=False, linear_op=None, **linear_opts):
    """
    :param int num_units:
    :param int depth: internal depth
    :param bool dense_input_transform:
    :param BlocksparseLinear linear_op:
    :param int seed:
    :param int block_size:
    :param int connectivity:
    """
    assert depth >= 2
    super(BlocksparseMultiplicativeMultistepLSTMCell, self).__init__()
    self.num_units = num_units
    self.depth = depth
    self.dense_input_transform = dense_input_transform
    if linear_op:
      self.linear = linear_op
    else:
      self.linear = BlocksparseLinear(**linear_opts)

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(c=self.num_units, h=self.num_units)

  def get_input_transformed(self, x):
    """
    :param tf.Tensor x:
    :rtype: (tf.Tensor, tf.Tensor)
    """
    with tf.variable_scope('input'):
      with tf.variable_scope('x1'):
        dim = self.num_units
        x1 = self.linear(x, dim, dense=self.dense_input_transform)
        x1.set_shape((None, None, dim))  # (time,batch,dim)
      with tf.variable_scope('x2'):
        dim = self.num_units
        x2 = self.linear(x, dim, dense=self.dense_input_transform)
        x2.set_shape((None, None, dim))  # (time,batch,dim)
      return x1, x2

  # noinspection PyMethodOverriding
  def call(self, inputs, state):
    """
    :param (tf.Tensor,tf.Tensor) inputs: each (batch, num_units)
    :param rnn_cell.LSTMStateTuple state: (batch, num_units)
    :return: output, new_state
    """
    _x1, _x2 = inputs
    assert isinstance(_x1, tf.Tensor) and isinstance(_x2, tf.Tensor), "inputs %r unexpected" % (inputs,)
    assert isinstance(state, rnn_cell.LSTMStateTuple)
    # All internal steps performed with moved feature axis, should be faster.
    x1 = self.linear.move_feature_axis(_x1, old_axis=-1, new_axis=self.linear.mul_feature_axis)
    x2 = self.linear.move_feature_axis(_x2, old_axis=-1, new_axis=self.linear.mul_feature_axis)
    h = self.linear.move_feature_axis(state.h, old_axis=-1, new_axis=self.linear.mul_feature_axis)
    c = self.linear.move_feature_axis(state.c, old_axis=-1, new_axis=self.linear.mul_feature_axis)

    with tf.variable_scope("step0"):
      dim = self.num_units
      h = self.linear(h, dim, with_bias=False, feature_axis=self.linear.mul_feature_axis)
      h *= x1
    with tf.variable_scope("step1"):
      h = self.linear(h, dim, with_bias=False, feature_axis=self.linear.mul_feature_axis)
      h += x2
      h = tf.nn.relu(h)

    for step in range(2, self.depth):
      with tf.variable_scope('step%i' % step):
        dim = self.num_units
        h = self.linear(h, dim, feature_axis=self.linear.mul_feature_axis)
        h = tf.nn.relu(h)
        h.set_shape((dim if self.linear.mul_feature_axis == 0 else None, None))

    with tf.variable_scope("gating"):
      with tf.variable_scope("cell_in"):
        cell_in = self.linear(h, self.num_units, feature_axis=self.linear.mul_feature_axis)
        cell_in = tf.tanh(cell_in)
      with tf.variable_scope("gate_in"):
        gate_in = self.linear(h, self.num_units, feature_axis=self.linear.mul_feature_axis)
        gate_in = tf.sigmoid(gate_in)
      with tf.variable_scope("gate_forget"):
        gate_forget = self.linear(h, self.num_units, feature_axis=self.linear.mul_feature_axis)
        gate_forget = tf.sigmoid(gate_forget)
      with tf.variable_scope("gate_out"):
        gate_out = self.linear(h, self.num_units, feature_axis=self.linear.mul_feature_axis)
        gate_out = tf.sigmoid(gate_out)
      cell = c * gate_forget + cell_in * gate_in
      out = tf.tanh(cell) * gate_out

    # Move feature axis back to where it is expected.
    cell = self.linear.move_feature_axis(cell, old_axis=self.linear.mul_feature_axis, new_axis=-1)
    out = self.linear.move_feature_axis(out, old_axis=self.linear.mul_feature_axis, new_axis=-1)
    cell.set_shape((None, self.num_units))
    out.set_shape((None, self.num_units))
    assert out.get_shape().dims[0].value == cell.get_shape().dims[0].value == _x1.get_shape().dims[0].value, 'b.dim'
    return out, rnn_cell.LSTMStateTuple(c=cell, h=out)
