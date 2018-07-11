#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("PersistenceVector")
    .Input("persistence_diagram: float")
    .Input("cardinalities: float")
    .Input("gaussians: float")
    .Input("nu: float")
    .Output("persistence_vector: float");

class PersistenceVectorOp : public OpKernel {
 public:

  explicit PersistenceVectorOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    DCHECK_EQ(context->num_inputs(), 4);

    const Tensor & diagram   = context->input(0);
    const Tensor & cards     = context->input(1);
    const Tensor & gaussians = context->input(2);
    float nu                 = context->input(3).matrix<float>()(0,0);

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
        float mu_0           = gaussians_tensor(g,0);
        float mu_1           = gaussians_tensor(g,1);
        float sigma_0_square = gaussians_tensor(g,2)*gaussians_tensor(g,2);
        float sigma_1_square = gaussians_tensor(g,3)*gaussians_tensor(g,3);
        for(int i = 0; i < num_pts_in_diagram; i++){
          float p0 = diagram_tensor(current + i,0);
          float p1 = diagram_tensor(current + i,1);
          float x_0 = (p1+p0)/sqrt(2);
          float x_1 = (p1-p0)/sqrt(2);
          float dist0 = x_0 - mu_0; float dist0_square = dist0*dist0;
          float dist1 = x_1 - mu_1; float dist1_square = dist1*dist1;
          float dist1_log = nu*log(x_1/nu) + nu - mu_1; float dist1_log_square = dist1_log*dist1_log;
          if(x_1 != 0){
            if(x_1 >= nu) output_tensor(instance, g) += exp(  -sigma_0_square*dist0_square  -sigma_1_square*dist1_square      );
            else          output_tensor(instance, g) += exp(  -sigma_0_square*dist0_square  -sigma_1_square*dist1_log_square  );
          }
        }
      }

      current += num_pts_in_diagram;

    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PersistenceVector").Device(DEVICE_CPU), PersistenceVectorOp);


