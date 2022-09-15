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

#include "lite/kernels/xpu/__xpu__rope_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

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

void XPURopeCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto cos_ptr = param.cos->data<float>();
  auto sin_ptr = param.sin->data<float>();

  std::cout << "rope preparerun, cos buffer size: " << param.cos->memory_size() << ", dim" << param.cos->dims() 
            << ", cos ptr " << cos_ptr << ", numel " << param.cos->numel() << std::endl;
  std::cout << "rope preparerun, sin buffer size: " << param.sin->memory_size() << ", dim" << param.sin->dims()
            << ", sin ptr " << sin_ptr << ", numel " << param.sin->numel() << std::endl;
  // cos_buffer_ = TargetWrapperXPU::MallocScratchPad(param.cos->memory_size());
  // sin_buffer_ = TargetWrapperXPU::MallocScratchPad(param.sin->memory_size());

  // std::cout << "debug 123: " << param.cos->IsInitialized() << ", "<< param.cos->data<float>()[0]<< std::endl;
  // for (int i = 0; i < 26; i ++) {
  //   std::cout << cos_ptr[20 * 26 + i] << " ";
  //   if (i == 12) {
  //     std::cout << std::endl;
  //   }
  // }
  // auto mean = compute_mean<float>(cos_ptr, param.cos->numel());
  // std::cout << "mean: " << mean << std::endl;
  // auto std = compute_standard_deviation<float>(cos_ptr, param.cos->numel(), true, mean);
  // auto agr = compute_average_grow_rate<float>(cos_ptr, param.cos->numel());
  // std::cout << "cos one samples: mean is " << mean << ", std is " << std << ", agr " << agr << ", numel " << param.cos->numel()  << std::endl;

  
  // mean = compute_mean<float>(sin_ptr, param.sin->numel());
  // std = compute_standard_deviation<float>(sin_ptr, param.sin->numel(), true, mean);
  // agr = compute_average_grow_rate<float>(sin_ptr, param.sin->numel());
  // std::cout << "\nsin one samples: mean is " << mean << ", std is " << std << ", agr " << agr << ", numel " << param.sin->numel()  << std::endl;
  // for (int i = 0; i < 26; i ++) {
  //   std::cout << sin_ptr[20 * 26 + i] << " ";
  //   if (i == 12) {
  //     std::cout << std::endl;
  //   }
  // }
  // std::cout << std::endl;
  // lite::TargetWrapperXPU::MemcpySync(cos_buffer_->addr_,
  //                                     param.cos->data<float>(),
  //                                     param.cos->memory_size(),
  //                                     IoDirection::HtoD);
  // lite::TargetWrapperXPU::MemcpySync(sin_buffer_->addr_,
  //                                   param.sin->data<float>(),
  //                                   param.sin->memory_size(),
  //                                   IoDirection::HtoD);
}
void XPURopeCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  // std::cout << "xpu rope run: " << std::endl;
  std::vector<int> lod = {0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560};
  xdnn::VectorParam<int> xpu_lod = {lod.data(), lod.size(), nullptr};
  int r = xdnn::rope(ctx.GetRawContext(),    /* context */
                     param.input->mutable_data<float>(), /* input */
                    //  reinterpret_cast<const float*>(cos_buffer_->addr_), /* cos */
                    //  reinterpret_cast<const float*>(sin_buffer_->addr_), /* sin */
                    param.output->mutable_data<float>(TARGET(kXPU)),
                    param.cos->data<float>(),
                    param.sin->data<float>(),
                     12, 26, 12 * 26, 10, xpu_lod);
                    //  param.output->mutable_data<float>(TARGET(kXPU)) /* out */);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__rope,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPURopeCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Cos", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Sin", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
