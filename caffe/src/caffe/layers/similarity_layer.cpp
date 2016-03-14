/*
 *
 * @brief this is for generic verison of 
 *  computing simiarity between instance and proto types
 *  in case of CEDL, there is two types of proto
 *  one is for category proto 
 *   and the other is for attribute 
 *  good luck implemented by gtlim 2015.9.24
 *
 *  please synchronize cpp and cu version at the same time.
 *  this layer is highly optimizied by using cuda implementation
 * 
 */
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/similarity_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
 
template <typename Dtype>
void SimilarityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

   CHECK_EQ(bottom[0]->channels(),bottom[1]->channels()) <<" Dimension is not compatible ";  
   //Figure out the dimensions.
   N_ = bottom[1]->num();	 // number of proto_types 
   K_ = (bottom[1]->count())/N_; // dimension if proto
   M_ = bottom[0]->num();        // batch_size 

   //output is result of computation of similarity. 
   vector<int> top_shape = bottom[0]->shape();
   top_shape.resize(2);
   top_shape[1] = N_;
   top[0]->Reshape(top_shape);

}

template <typename Dtype>
void SimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* img_feat = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* proto  = bottom[1]->cpu_data();
  //computing similarity between proto and img_feat
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
     img_feat, proto , (Dtype)0., top_data);
}
template <typename Dtype>
void SimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
     //Implementation of CEDL
     //backprop to img_feat
     caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
         top_diff, bottom[1]->cpu_data(), (Dtype)0.,
         bottom[0]->mutable_cpu_diff()); 
  }
  if (propagate_down[1] ) {
    const Dtype* top_diff = top[0]->cpu_diff();
     //backprop to proto type
     caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
         top_diff, bottom[0]->cpu_data(), (Dtype)0.,
         bottom[1]->mutable_cpu_diff());   
   }
}
#ifdef CPU_ONLY
STUB_GPU(SimilarityLayer);
#endif

INSTANTIATE_CLASS(SimilarityLayer);
REGISTER_LAYER_CLASS(Similarity);

}  // namespace caffe
