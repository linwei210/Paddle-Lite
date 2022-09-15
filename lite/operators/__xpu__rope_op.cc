// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/operators/__xpu__rope_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPURopeOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.cos);
  CHECK_OR_FALSE(param_.sin);

  const auto input_dims = param_.input->dims();
  const auto cos_dims = param_.cos->dims();
  const auto sin_dims = param_.sin->dims();
  CHECK_EQ_OR_FALSE(input_dims.size(), 4UL);
  CHECK_EQ_OR_FALSE(cos_dims.size(), 4UL);
  CHECK_EQ_OR_FALSE(sin_dims.size(), 4UL);

  int64_t cos_max_length = cos_dims[2];
  int64_t sin_max_length = sin_dims[2];
  int64_t cos_head_dim = cos_dims[3];
  int64_t sin_head_dim = sin_dims[3];
  int64_t input_head_dim = input_dims[3];
  int64_t input_seq_length = input_dims[1];
  CHECK_EQ_OR_FALSE(cos_max_length, sin_max_length);
  CHECK_EQ_OR_FALSE(cos_head_dim, sin_head_dim);
  CHECK_EQ_OR_FALSE(input_head_dim, cos_head_dim);
  CHECK_GE_OR_FALSE(cos_max_length, input_seq_length);

  return true;
}

bool XPURopeOp::InferShapeImpl() const {
  auto input_dims = param_.input->dims();
  // Set output dims
  param_.output->Resize(input_dims);
  // std::vector<DDim::value_type> output_dims(input_dims.size());
  // std::cout << "XPURopeOp output dim: ";
  // for (int i = 0; i < input_dims.size(); ++i) {
  //   std::cout << input_dims[i] << " ";
  //   output_dims[i] = input_dims[i];
  // }
  // std::cout << std::endl;
  // param_.output->Resize(output_dims);
  // share LoD
  param_.output->set_lod(param_.input->lod());

  return true;
}

  template <typename T>
  double compute_mean(const T* in, const size_t length) {
    double sum = 0.;
    for (size_t i = 0; i < length; ++i) {
      sum += in[i];
    }
    return sum / length;
  }

  template <typename T>
  double compute_standard_deviation(const T* in,
                                    const size_t length,
                                    bool has_mean = false,
                                    double mean = 10000) {
    if (!has_mean) {
      mean = compute_mean<T>(in, length);
    }

    double variance = 0.;
    for (size_t i = 0; i < length; ++i) {
      variance += pow((in[i] - mean), 2);
    }
    variance /= length;
    return sqrt(variance);
  }

      template <typename T>
  double compute_average_grow_rate(const T* in, const size_t length) {
    const double eps = 1e-5;
    double ave_grow_rate = 0.0f;
    for (size_t i = 1; i < length; ++i) {
      ave_grow_rate += (in[i] - in[i - 1]) / (in[i - 1] + eps);
    }
    ave_grow_rate /= length;
    return ave_grow_rate;
  }

bool XPURopeOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Input("Cos").front()));
  CHECK(scope->FindVar(op_desc.Input("Sin").front()));
  CHECK(scope->FindVar(op_desc.Output("Output").front()));

  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  // param_.cos =
  //     scope->FindVar(op_desc.Input("Cos").front())->GetMutable<Tensor>();
  // param_.sin =
  //     scope->FindVar(op_desc.Input("Sin").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();
  // auto input_name = op_desc.Input("Input").front();
  // param_.input = const_cast<lite::Tensor*>(&scope->FindVar(input_name)->Get<lite::Tensor>());
  auto cos_name = op_desc.Input("Cos").front();
  param_.cos = const_cast<lite::Tensor*>(&scope->FindVar(cos_name)->Get<lite::Tensor>());
  auto sin_name = op_desc.Input("Sin").front();
  param_.sin = const_cast<lite::Tensor*>(&scope->FindVar(sin_name)->Get<lite::Tensor>());
    auto input_name = op_desc.Input("Input").front();

  static bool print_buffer = true;

  if (print_buffer) {
    print_buffer = false;
  auto cos_ptr = param_.cos->data<float>();
  auto mean = compute_mean<float>(cos_ptr, param_.cos->numel());
  auto std = compute_standard_deviation<float>(cos_ptr, param_.cos->numel(), true, mean);
  auto agr = compute_average_grow_rate<float>(cos_ptr, param_.cos->numel());
  std::cout << "cos one samples: mean is " << mean << ", std is " << std << ", agr " << agr << ", numel " << param_.cos->numel() << std::endl;
  for (int i = 0; i < 26; i ++) {
    std::cout << cos_ptr[20 * 26 + i] << " ";
    if (i == 12) {
      std::cout << std::endl;
    }
  }
  auto sin_ptr = param_.sin->data<float>();
  mean = compute_mean<float>(sin_ptr, param_.sin->numel());
  std = compute_standard_deviation<float>(sin_ptr, param_.sin->numel(), true, mean);
  agr = compute_average_grow_rate<float>(sin_ptr, param_.sin->numel());
  std::cout << "\nsin one samples: mean is " << mean << ", std is " << std << ", agr " << agr << ", numel " << param_.sin->numel() << std::endl;
  for (int i = 0; i < 26; i ++) {
    std::cout << sin_ptr[20 * 26 + i] << " ";
    if (i == 12) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__rope, paddle::lite::operators::XPURopeOp);
