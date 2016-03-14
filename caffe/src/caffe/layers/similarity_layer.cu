
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/similarity_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 //implementation of cuda (using cblas highly optimized for matrix computation)
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
       bottom[0]->gpu_data(),bottom[1]->gpu_data(), (Dtype)0., top_data);
}

template <typename Dtype>
void SimilarityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
   const Dtype* top_diff = top[0]->gpu_diff();
   //backprop to img_feat 
   caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
      top_diff, bottom[1]->gpu_data(), (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
   const Dtype* top_diff = top[0]->gpu_diff();
   //backprop to proto type
   caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom[0]->gpu_data(), (Dtype)0.,bottom[1]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SimilarityLayer);

}  // namespace caffe
