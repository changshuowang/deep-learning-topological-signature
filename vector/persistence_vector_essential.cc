#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("PersistenceVectorEssential")
    .Input("persistence_diagram: float")
    .Input("cardinalities: float")
    .Input("gaussians: float")
    .Output("persistence_vector_essential: float");

class PersistenceVectorEssentialOp : public OpKernel {
 public:

  explicit PersistenceVectorEssentialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    DCHECK_EQ(context->num_inputs(), 3);

    const Tensor & diagram   = context->input(0);
    const Tensor & cards     = context->input(1);
    const Tensor & gaussians = context->input(2);

    const TensorShape & gaussians_shape = gaussians.shape();
    const int num_pts_gaussians         = gaussians_shape.dim_size(0);
    const TensorShape & cards_shape     = cards.shape();
    const int batch_size                = cards_shape.dim_size(0);

    TensorShape output_shape;
    output_shape.AddDim(batch_size);
    output_shape.AddDim(num_pts_gaussians);

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto diagram_tensor     = diagram.matrix<float>();
    auto cards_tensor       = cards.matrix<float>();
    auto gaussians_tensor   = gaussians.matrix<float>();
    auto output_tensor      = output->matrix<float>();

    int current = 0;
    for(int instance = 0; instance < batch_size; instance++){

      int num_pts_in_diagram = cards_tensor(instance,0);

      for(int g = 0; g < num_pts_gaussians; g++){
        output_tensor(instance, g) = 0;
        float mu           = gaussians_tensor(g,0);
        float sigma_square = gaussians_tensor(g,1)*gaussians_tensor(g,1);
        for(int i = 0; i < num_pts_in_diagram; i++){
          float p = diagram_tensor(current + i,0);
          float dist = p - mu; float dist_square = dist*dist;
          output_tensor(instance, g) += exp(  -sigma_square*dist_square      );
        }
      }

      current += num_pts_in_diagram;

    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PersistenceVectorEssential").Device(DEVICE_CPU), PersistenceVectorEssentialOp);


