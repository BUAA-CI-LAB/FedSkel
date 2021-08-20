#include <vector>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include "caffe/layers/conv_layer.hpp"
using namespace std ; 

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  this->this_iter +=1 ;
  struct timeval time1, time2, time3, time4;

  // gettimeofday(&time1, NULL);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
  // gettimeofday(&time2, NULL);
  // double forward = ((time2.tv_sec - time1.tv_sec) * 1000000 + (time2.tv_usec - time1.tv_usec)) / float(1000000) ;
  // cout << "forward: " << forward << endl; 
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    struct timeval time1, time2, time3, time4;
    double pruning=0, gta=0, gtw = 0; 
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        gettimeofday(&time1, NULL);
        if(this->this_iter >= this->setskel_iter && this->skel_rate > 0) this->prune_gradients(top_diff + n * this->top_dim_) ;
        gettimeofday(&time2, NULL);

        if (this->param_propagate_down_[0]) {
          if(this->this_iter >= this->setskel_iter && this->skel_rate > 0){
            this->weight_cpu_gemm_fedskel(bottom_data + n * this->bottom_dim_,
                                     top_diff + n * this->top_dim_, weight_diff);
          }
          else{
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                  top_diff + n * this->top_dim_, weight_diff);
          }
        }
        gettimeofday(&time3, NULL);

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if(this->this_iter >= this->setskel_iter && this->skel_rate >0){
            this->backward_cpu_gemm_fedskel(top_diff + n * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
          }
          else{
            this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
