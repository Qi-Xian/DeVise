#ifndef CAFFE_RANKING_LOSS_LAYER_HPP_
#define CAFFE_RANKING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class RankingLossLayer : public LossLayer<Dtype> {
 public:
  explicit RankingLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RankingLoss"; }

 protected:
  /// @copydoc RankingLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
 //This is for cuda implementation
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype margin;
  bool RankingLossParameter_Trunc;
};

}  // namespace caffe

#endif  // CAFFE_RANKING_LOSS_LAYER_HPP_
