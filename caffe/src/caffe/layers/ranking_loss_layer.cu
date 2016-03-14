#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/ranking_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RankingLossForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, Dtype margin) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;    //row
    const int label_value = static_cast<int>(label[n]);
    const int label_index = n*dim + label_value;
    if( label_index != index ) {
     loss[index] = max( Dtype(0),
	margin - bottom_data[label_index] + bottom_data[index] );
    } else {
     loss[index] = 0;
    }
  }
}
template <typename Dtype>
void RankingLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count()/num;
  const int nthreads = num * dim; 
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  RankingLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, label, loss_data,
      num, dim , margin); 
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= num;
  top[0]->mutable_cpu_data()[0] = loss;
}


//
//implementation of 1D Grid and 1D Blocks 
// TODO( measure cpu_time and improve performace)
template <typename Dtype>
__global__ void RankingLossBackwardGPU(  const Dtype* bottom_data,const Dtype* label, 
                    Dtype* bottom_diff, const int num, const int dim, int trunc, Dtype margin) {
 unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if(ix < num) { 
  // printf("%d %d %d\n",ix,trunc,dim);
   const int label_value = static_cast<int>(label[ix]); 
   const int label_index = dim*ix + label_value;
   for( int iy = 0 ; iy < dim ; iy++){
	const int index = dim*ix + iy;
	if( label_index != index ) {
         const Dtype prob = max( Dtype(0),
	  margin - bottom_data[label_index] + bottom_data[index] );
	 if( prob > 0){
	  bottom_diff[index] = 1;
	  bottom_diff[label_index] -= 1;
	  if(trunc) break; //this is for truncate version
	}
      }
    }     
  }
}
template <typename Dtype>
void RankingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int Trunc_ = this->layer_param_.ranking_loss_param().trunc();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype scale = top[0]->cpu_diff()[0]/num;
    //initalize bottom_diff to zero
    caffe_gpu_set(bottom[0]->count() , Dtype(0) , bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    //Checking most powerful performance
    dim3 block(32); dim3 grid((num + block.x - 1) / block.x);
    RankingLossBackwardGPU<Dtype><<<grid,
         block>>>( bottom_data, label, bottom_diff,
         num, dim, Trunc_,margin); 
    caffe_gpu_scal(bottom[0]->count(), scale , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RankingLossLayer);

}  // namespace caffe
