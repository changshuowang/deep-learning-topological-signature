import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

persistence_vector_essential_grad_module = tf.load_op_library('persistence_vector_essential_grad.so')

@ops.RegisterGradient("PersistenceVectorEssential")
def _persistence_vector_essential_grad_cc(op, grad):
    return persistence_vector_essential_grad_module.persistence_vector_essential_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])    