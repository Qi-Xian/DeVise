#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layers/ranking_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RankingLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  RankingLossParameter_Trunc = this->layer_param_.ranking_loss_param().trunc();
  margin = Dtype(this->layer_param_.ranking_loss_param().margin());  
}

template <typename Dtype>
void RankingLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  int label;
  for (int i = 0; i < num; ++i) {
     label = static_cast<int>(bottom_label[i]);
     for(int j = 0; j < dim; ++j) {
      if( j == label) continue;
    	Dtype prob = std::max( Dtype(0),
    	   margin - bottom_data[i * dim + label] + bottom_data[i*dim + j] );
	if(prob > 0) {
      	 loss += prob;
	if(RankingLossParameter_Trunc) 
 	 break;	// truncating the sum after ther first margin-violating false term was encountered
	  }
	 }
	}
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void RankingLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int count = bottom[0]->count();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = top[0]->cpu_diff()[0]/num;
    int label;
    for (int i = 0; i < num; ++i) {
      label = static_cast<int>(bottom_label[i]);
      for(int j = 0; j < dim; ++j) {
       if( j == label) continue;
    	Dtype prob = std::max( Dtype(0),
    	 margin - bottom_data[i * dim + label] + bottom_data[i*dim + j] );
       if( prob > 0) {
	bottom_diff[i*dim + j] = 1;
	bottom_diff[i*dim + label] -= 1;
       if(RankingLossParameter_Trunc) 
	break;// truncating the sum after ther first margin-violating false term was encountered
       }
      }  
    }
  caffe_scal(count, scale, bottom_diff);
 }
}

INSTANTIATE_CLASS(RankingLossLayer);
REGISTER_LAYER_CLASS(RankingLoss);

}  // namespace caffe
