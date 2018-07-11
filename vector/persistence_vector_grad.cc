#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <math.h>
#include <stdio.h>


using namespace tensorflow;

REGISTER_OP("PersistenceVectorGrad")
  .Input("grad: float")
  .Input("persistence_diagram: float")
  .Input("cardinalities: float")
  .Input("gaussians: float")
  .Input("nu: float")
  .Output("grad_persistence_diagram: float")
  .Output("grad_cards: float")
  .Output("grad_gaussians: float")
  .Output("grad_nu: float");

class PersistenceVectorGradOp : public OpKernel {
public:

  explicit PersistenceVectorGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    DCHECK_EQ(5, context->num_inputs());

    const Tensor & grad      = context->input(0);
    const Tensor & diagram   = context->input(1);
    const Tensor & cards     = context->input(2);
    const Tensor & gaussians = context->input(3);
    float nu                 = context->input(4).matrix<float>()(0,0);

    const TensorShape & cards_shape        = cards.shape();
    const int batch_size                   = cards_shape.dim_size(0);
    const TensorShape & gaussians_shape    = gaussians.shape();
    const int num_pts_gaussians            = gaussians_shape.dim_size(0);
    const TensorShape & nu_shape           = context->input(3).shape();
    const TensorShape & diagram_shape      = diagram.shape();

    Tensor* grad_diagram   = NULL;
    Tensor* grad_cards     = NULL;
    Tensor* grad_gaussians = NULL;
    Tensor* grad_nu        = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, diagram_shape,   &grad_diagram));
    OP_REQUIRES_OK(context, context->allocate_output(1, cards_shape,     &grad_cards));
    OP_REQUIRES_OK(context, context->allocate_output(2, gaussians_shape, &grad_gaussians));
    OP_REQUIRES_OK(context, context->allocate_output(3, nu_shape,        &grad_nu));

    auto grad_tensor            = grad.matrix<float>();
    auto gaussians_tensor       = gaussians.matrix<float>();
    auto diagram_tensor         = diagram.matrix<float>();
    auto cards_tensor           = cards.matrix<float>();
    auto grad_diagram_tensor    = grad_diagram->matrix<float>();
    auto grad_gaussians_tensor  = grad_gaussians->matrix<float>();

    for(int g = 0; g < num_pts_gaussians; g++){

      grad_gaussians_tensor(g,0) = 0;
      grad_gaussians_tensor(g,1) = 0;
      grad_gaussians_tensor(g,2) = 0;
      grad_gaussians_tensor(g,3) = 0;

      float mu_0           = gaussians_tensor(g,0);
      float mu_1           = gaussians_tensor(g,1);
      float sigma_0        = gaussians_tensor(g,2);
      float sigma_1        = gaussians_tensor(g,3);
      float sigma_0_square = sigma_0*sigma_0;
      float sigma_1_square = sigma_1*sigma_1;

      int current = 0;
      for(int instance = 0; instance < batch_size; instance++){

        int num_pts_in_diagram = cards_tensor(instance,0);
        float q0, q1, q2, q3; q0 = 0; q1 = 0; q2 = 0; q3 = 0;

        for(int j = 0; j < num_pts_in_diagram; j++){

          float p0 = diagram_tensor(current+j,0);
          float p1 = diagram_tensor(current+j,1);
          float x_1 = (p1-p0)/sqrt(2);

          if(x_1 != 0){

            float x_0 = (p1+p0)/sqrt(2);
            float dist0 = x_0 - mu_0; float dist0_square = dist0*dist0;
            float dist1 = x_1 - mu_1; float dist1_square = dist1*dist1;
            float dist1_log = nu*log(x_1/nu) + nu - mu_1; float dist1_log_square = dist1_log*dist1_log;

            if(x_1 >= nu){
              float function_value = exp(  -sigma_0_square*dist0_square  -sigma_1_square*dist1_square      );
              q0 += function_value*(2*sigma_0_square*dist0);
              q1 += function_value*(2*sigma_1_square*dist1);
              q2 += function_value*(-2*dist0_square*sigma_0);
              q3 += function_value*(-2*dist1_square*sigma_1);
            }
            else{
              float function_value = exp(  -sigma_0_square*dist0_square  -sigma_1_square*dist1_log_square  );
              q0 += function_value*(2*sigma_0_square*dist0);
              q1 += function_value*(2*sigma_1_square*dist1_log);
              q2 += function_value*(-2*dist0_square*sigma_0);
              q3 += function_value*(-2*dist1_log_square*sigma_1);
            }
          }
        }

        grad_gaussians_tensor(g,0) += grad_tensor(instance,g)*q0;
        grad_gaussians_tensor(g,1) += grad_tensor(instance,g)*q1;
        grad_gaussians_tensor(g,2) += grad_tensor(instance,g)*q2;
        grad_gaussians_tensor(g,3) += grad_tensor(instance,g)*q3;

        current += num_pts_in_diagram;

      }      
    }


    int current = 0;
    for(int instance = 0; instance < batch_size; instance++){
     
      int num_pts_in_diagram = cards_tensor(instance,0);

      for(int j = 0; j < num_pts_in_diagram; j++){

        grad_diagram_tensor(current+j, 0) = 0; grad_diagram_tensor(current+j, 1) = 0;
        float x_1 = (  -diagram_tensor(current+j, 0) + diagram_tensor(current+j, 1)  )/sqrt(2);

        if(x_1 != 0){

          float x_0 = (  diagram_tensor(current+j, 0)  + diagram_tensor(current+j, 1)  )/sqrt(2);

          for(int i = 0; i < num_pts_gaussians; i++){

            float mu_0             = gaussians_tensor(i,0);
            float mu_1             = gaussians_tensor(i,1);
            float sigma_0          = gaussians_tensor(i,2);
            float sigma_0_square   = sigma_0*sigma_0;
            float sigma_1          = gaussians_tensor(i,3);
            float sigma_1_square   = sigma_1*sigma_1;
            float dist0            = x_0 - mu_0;
            float dist0_square     = dist0*dist0;
            float dist1            = x_1 - mu_1;
            float dist1_square     = dist1*dist1;
            float dist1_log        = nu*log(x_1/nu) + nu - mu_1;
            float dist1_log_square = dist1_log*dist1_log;

            if(x_1 >= nu){
              float function_value = exp(  -sigma_0_square*dist0_square  -sigma_1_square*dist1_square      );
              grad_diagram_tensor(current+j, 0) += grad_tensor(instance,i)*function_value*2/sqrt(2)*(-sigma_0_square*dist0 + sigma_1_square*dist1);
              grad_diagram_tensor(current+j, 1) += grad_tensor(instance,i)*function_value*2/sqrt(2)*(-sigma_0_square*dist0 - sigma_1_square*dist1);
            }
            else{
              float function_value = exp(  -sigma_0_square*dist0_square  -sigma_1_square*dist1_log_square      );
              grad_diagram_tensor(current+j, 0) += grad_tensor(instance,i)*function_value*2/sqrt(2)*(-sigma_0_square*dist0 + sigma_1_square*dist1_log*nu/x_1);
              grad_diagram_tensor(current+j, 1) += grad_tensor(instance,i)*function_value*2/sqrt(2)*(-sigma_0_square*dist0 - sigma_1_square*dist1_log*nu/x_1);
            }
          
          }
        }

      }

      current += num_pts_in_diagram;
    }


  }
};

REGISTER_KERNEL_BUILDER(Name("PersistenceVectorGrad").Device(DEVICE_CPU), PersistenceVectorGradOp);
