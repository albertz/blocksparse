
import os
import tensorflow as tf


def get_module():
  from TFUtil import OpCodeCompiler

  cc_files = [
    "batch_norm_op.cc",
    "blocksparse_l2_norm_op.cc", "blocksparse_matmul_op.cc",
    "cwise_linear_op.cc", "edge_bias_op.cc", "ew_op.cc", "gpu_types.cc", "layer_norm_op.cc"
  ]
  # not "blocksparse_kernels.cc" (SASS kernels) for now.
  # "blocksparse_conv_op.cc" depends on that.

  cu_files = [
    "batch_norm_op_gpu.cu", "blocksparse_l2_norm_op_gpu.cu", "blocksparse_matmul_op_gpu.cu",
    "cwise_linear_op_gpu.cu", "edge_bias_op_gpu.cu", "ew_op_gpu.cu",
    "layer_norm_cn_op_gpu.cu", "layer_norm_nc_op_gpu.cu"
  ]

  src_dir = os.path.normpath(os.path.dirname(__file__) + "/../src")
  assert os.path.isdir(src_dir)
  src_code = ""
  for fn in cc_files + cu_files:
    src_code += "\n// ----- %s BEGIN -----\n" % fn
    src_code += open("%s/%s" % (src_dir, fn)).read()
    src_code += "\n// ----- %s END -----\n\n\n" % fn

  src_code += """
  // custom code
  Status GetKernel(std::string& kernel_name, CUfunction* kernel) {
    if(*kernel) return Status::OK();  // Only need to get kernel once.
    return errors::Internal("Blocksparse GetKernel without SASS: kernel ", kernel_name, " not available");
  }
  """

  compiler = OpCodeCompiler(base_name="blocksparse", code=src_code, code_version=1, include_paths=[src_dir])
  return tf.load_op_library(compiler.get_lib_filename())
