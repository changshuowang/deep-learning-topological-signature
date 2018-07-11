#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <math.h>
#include <stdio.h>

using namespace tensorflow;

REGISTER_OP("PersistenceVectorEssentialGrad")
  .Input("grad: float")
  .Input("persistence_diagram: float")
  .Input("cardinalities: float")
  .Input("gaussians: float")
  .Output("grad_diagram: float")
  .Output("grad_cards: float")
  .Output("grad_gaussians: float");

class PersistenceVectorEssentialGradOp : public OpKernel {
public:

  explicit PersistenceVectorEssentialGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

      DCHECK_EQ(4, context->num_inputs());

      const Tensor & grad      = context->input(0);
      const Tensor & diagram   = context->input(1);
      const Tensor & cards     = context->input(2);
      const Tensor & gaussians = context->input(3);

      const TensorShape & cards_shape        = cards.shape();
      const int batch_size                   = cards_shape.dim_size(0);
      const TensorShape & gaussians_shape    = gaussians.shape();
      const int num_pts_gaussians            = gaussians_shape.dim_size(0);
      const TensorShape & diagram_shape      = diagram.shape();

      Tensor* grad_diagram   = NULL;
      Tensor* grad_cards     = NULL;
      Tensor* grad_gaussians = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, diagram_shape,   &grad_diagram));
      OP_REQUIRES_OK(context, context->allocate_output(1, cards_shape,     &grad_cards));
      OP_REQUIRES_OK(context, context->allocate_output(2, gaussians_shape, &grad_gaussians));

      auto grad_tensor            = grad.matrix<float>();
      auto gaussians_tensor       = gaussians.matrix<float>();
      auto diagram_tensor         = diagram.matrix<float>();
      auto cards_tensor           = cards.matrix<float>();
      auto grad_diagram_tensor    = grad_diagram->matrix<float>();
      auto grad_gaussians_tensor  = grad_gaussians->matrix<float>();

      for(int g = 0; g < num_pts_gaussians; g++){

        grad_gaussians_tensor(g,0) = 0;
        grad_gaussians_tensor(g,1) = 0;

        float mu           = gaussians_tensor(g,0);
        float sigma        = gaussians_tensor(g,1);
        float sigma_square = sigma*sigma;

        int current = 0;
        for(int instance = 0; instance < batch_size; instance++){

          int num_pts_in_diagram = cards_tensor(instance,0);
          float q0, q1; q0 = 0; q1 = 0;

          for(int j = 0; j < num_pts_in_diagram; j++){

            float p = diagram_tensor(current+j,0);
            float dist = p - mu; float dist_square = dist*dist;
            float function_value = exp(  -sigma_square*dist_square  );
            q0 += function_value*(2*sigma_square*dist);
            q1 += function_value*(-2*dist_square*sigma);

          }
          grad_gaussians_tensor(g,0) += grad_tensor(instance,g)*q0;
          grad_gaussians_tensor(g,1) += grad_tensor(instance,g)*q1;

          current += num_pts_in_diagram;

        }
      }


    int current = 0;
    for(int instance = 0; instance < batch_size; instance++){

      int num_pts_in_diagram = cards_tensor(instance,0);

      for(int j = 0; j < num_pts_in_diagram; j++){

        grad_diagram_tensor(current+j,0) = 0;

        for(int i = 0; i < num_pts_gaussians; i++){

          float mu             = gaussians_tensor(i,0);
          float sigma          = gaussians_tensor(i,1);
          float sigma_square   = sigma*sigma;
          float dist           = diagram_tensor(current+j,0) - mu;
          float dist_square    = dist*dist;

          float function_value = exp(-sigma_square*dist_square);
          grad_diagram_tensor(current+j,0) += grad_tensor(instance,i)*function_value*2*(-sigma_square*dist);

        }
      }

      current += num_pts_in_diagram;
    }
  



  }
};

REGISTER_KERNEL_BUILDER(Name("PersistenceVectorEssentialGrad").Device(DEVICE_CPU), PersistenceVectorEssentialGradOp);
