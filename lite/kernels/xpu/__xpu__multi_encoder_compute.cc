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

#include "lite/kernels/xpu/__xpu__multi_encoder_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
static std::vector<const T*> prepare_weight(
    const std::vector<lite::Tensor*>& fc_weight) {
  std::vector<const T*> result;
  for (auto* weight : fc_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}

template <typename T>
std::vector<const T*>* XPUMultiEncoderCompute::get_weight() {
  LOG(FATAL) << "Invalid Weight Type";
  return nullptr;
}

template <>
std::vector<const int16_t*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_int16_;
}

template <>
std::vector<const int8_t*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_int8_;
}

template <>
std::vector<const float*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_fp32_;
}

template <>
std::vector<const float16*>* XPUMultiEncoderCompute::get_weight() {
  return &arg_fc_weight_fp16_;
}

void XPUMultiEncoderCompute::prepare_quant_max(
    const std::vector<float>& max_value,
    int n_layers,
    int max_ptr_len,
    std::vector<const float*>& max_xpu_ptrs) {
  bool mul_quant = false;
  bool matmul_quant = false;
  if (max_value.size() == (n_layers * 12)) {
    mul_quant = true;
  } else if (max_value.size() == (n_layers * 18)) {
    mul_quant = true;
    matmul_quant = true;
  } else if (max_value.size() == 0) {
    // dynamic quant, find max in xpu
    return;
  } else {
    LOG(FATAL) << "invalid quant max value for xpu encoder, "
               << max_value.size() << ", n_layers: " << n_layers;
  }
  // prepare input_max
  input_max_guard_ = TargetWrapperXPU::MallocScratchPad(
      max_value.size() * max_ptr_len * sizeof(float));
  float* input_max_ptr = reinterpret_cast<float*>(input_max_guard_->addr_);
  std::vector<float> cpu_max;
  cpu_max.resize(max_value.size() * max_ptr_len);
  for (int i = 0; i < max_value.size(); ++i) {
    for (int j = 0; j < max_ptr_len; ++j) {
      cpu_max[i * max_ptr_len + j] = max_value[i];
    }
  }
  lite::TargetWrapperXPU::MemcpySync(
      input_max_ptr,
      cpu_max.data(),
      sizeof(float) * max_ptr_len * max_value.size(),
      IoDirection::HtoD);
  max_xpu_ptrs.resize(max_value.size());
  for (int i = 0; i < max_value.size(); i += 1) {
    max_xpu_ptrs[i] = input_max_ptr + i * max_ptr_len;
  }
  return;
}

void XPUMultiEncoderCompute::prepare_weight_max(
    bool per_channel,
    const std::vector<lite::Tensor*>& weight_max,
    int max_ptr_len,
    std::vector<const float*>& max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : weight_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;

  if (!per_channel) {
    max_value_num *= max_ptr_len;
  }
  weight_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* weight_max_ptr = reinterpret_cast<float*>(weight_max_guard_->addr_);

  int offset = 0;
  for (auto max_tensor : weight_max) {
    float* cur_weight_max_ptr = weight_max_ptr + offset;
    auto len = max_tensor->numel();
    VLOG(6) << "weight max value: " << max_tensor->data<float>()[0] << " "
            << max_tensor->data<float>()[len - 1];
    if (per_channel) {
      lite::TargetWrapperXPU::MemcpySync(cur_weight_max_ptr,
                                         max_tensor->raw_data(),
                                         sizeof(float) * len,
                                         IoDirection::HtoD);
      max_xpu_ptrs.push_back(cur_weight_max_ptr);
      offset += len;
    } else {
      std::vector<float> cpu_max(max_ptr_len, max_tensor->data<float>()[0]);
      lite::TargetWrapperXPU::MemcpySync(cur_weight_max_ptr,
                                         cpu_max.data(),
                                         sizeof(float) * max_ptr_len,
                                         IoDirection::HtoD);
      max_xpu_ptrs.push_back(cur_weight_max_ptr);
      offset += max_ptr_len;
    }
  }
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

void XPUMultiEncoderCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();
  // prepare bias
  for (auto* fc_bias : param.fc_bias) {
    arg_fc_bias_.push_back(fc_bias->data<float>());
  }
  // prepare scale
  for (auto* ln_scale : param.ln_scale) {
    arg_ln_scale_.push_back(ln_scale->data<float>());
  }
  // prepare ln_bias
  for (auto* ln_bias : param.ln_bias) {
    arg_ln_bias_.push_back(ln_bias->data<float>());
  }
  // prepare weights
  local_quant_ =
      GetBoolFromEnv("XPU_LOCAL_QUANT") || lite::TargetWrapperXPU::local_quant;
  if (param.precision == "int16") {
    if (local_quant_) {
      arg_fc_weight_fp16_ = prepare_weight<float16>(param.fc_weight);
    } else {
      arg_fc_weight_int16_ = prepare_weight<int16_t>(param.fc_weight);
    }
  } else if (param.precision == "int8") {
    arg_fc_weight_int8_ = prepare_weight<int8_t>(param.fc_weight);
  } else if (param.precision == "int31") {
    arg_fc_weight_fp32_ = prepare_weight<float>(param.fc_weight);
  }
  const int n_layers = param.fc_weight.size() / 6;
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  prepare_weight_max(
      param.per_channel, param.weight_max, XPU_QUANT_SCALE_NUM, fc_weight_max_);
  // prepare quant max, mul&matmul input/output max
  prepare_quant_max(
      param.input_max, n_layers, XPU_QUANT_SCALE_NUM, fc_input_max_);
  // prepare quant type
  for (auto quant_type : param.quant_types) {
    if (quant_type == "enable_int8") {
      quant_types_.push_back(xdnn::QuantType::QUANT_INT8);
    } else if (quant_type == "enable_int16") {
      quant_types_.push_back(xdnn::QuantType::QUANT_INT16);
    } else {
      quant_types_.push_back(xdnn::QuantType::NOT_QUANT);
    }
  }
  // prepare act_type
  if (param.act_type == "gelu") {
    qkv_act = xdnn::Activation_t::GELU;
  } else if (param.act_type != "relu") {
    CHECK(false) << "Invalid QKV Activation Type: " << param.act_type;
  }
  // prepare with sice
  if ((param.slice_starts.size() > 0 && param.slice_starts[0] == 0) &&
      (param.slice_ends.size() > 0 && param.slice_ends[0] == 1) &&
      (param.slice_axes.size() > 0 && param.slice_axes[0] == 1)) {
    slice_idx = 0;
  }
  // prepare input_cast and output_cast guard_
  cast_in_guard_ = TargetWrapperXPU::MallocScratchPad(4 * 1024 * 1024);
  cast_out_guard_ = TargetWrapperXPU::MallocScratchPad(4 * 1024 * 1024);

  //cos
  if (param.cos && GetBoolFromEnv("XPU_ENABLE_ROPE_KERNEL")) {
    std::cout << "cos buffer size: " << param.cos->memory_size() << ", dim" << param.cos->dims() << std::endl;
    std::cout << "sin buffer size: " << param.sin->memory_size() << ", dim" << param.sin->dims() << std::endl;
    cos_buffer_ = TargetWrapperXPU::MallocScratchPad(param.cos->memory_size());
    sin_buffer_ = TargetWrapperXPU::MallocScratchPad(param.sin->memory_size());
    auto cos_ptr = param.cos->data<float>();
    auto mean = compute_mean<float>(cos_ptr, param.cos->numel());
    auto std = compute_standard_deviation<float>(cos_ptr, param.cos->numel(), true, mean);
    auto agr = compute_average_grow_rate<float>(cos_ptr, param.cos->numel());
    std::cout << "cos one samples: mean is " << mean << ", std is " << std << ", agr " << agr << ", numel " << param.cos->numel()  << std::endl;
    for (int i = 0; i < 26; i ++) {
      std::cout << cos_ptr[20 * 26 + i] << " ";
      if (i == 12) {
        std::cout << std::endl;
      }
    }
    auto sin_ptr = param.sin->data<float>();
    mean = compute_mean<float>(sin_ptr, param.sin->numel());
    std = compute_standard_deviation<float>(sin_ptr, param.sin->numel(), true, mean);
    agr = compute_average_grow_rate<float>(sin_ptr, param.sin->numel());
    std::cout << "\nsin one samples: mean is " << mean << ", std is " << std << ", agr " << agr << ", numel " << param.sin->numel()  << std::endl;
    for (int i = 0; i < 26; i ++) {
      std::cout << sin_ptr[20 * 26 + i] << " ";
      if (i == 12) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;

    lite::TargetWrapperXPU::MemcpySync(cos_buffer_->addr_,
                                        param.cos->data<float>(),
                                        param.cos->memory_size(),
                                        IoDirection::HtoD);
    lite::TargetWrapperXPU::MemcpySync(sin_buffer_->addr_,
                                      param.sin->data<float>(),
                                      param.sin->memory_size(),
                                      IoDirection::HtoD);
  }
}

template <typename T, typename TW, typename TGEMM>
void XPUMultiEncoderCompute::run_encoder(const T* in, T* out) {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const float* cos_buffer = nullptr;
  const float* sin_buffer = nullptr;
  if (param.cos && GetBoolFromEnv("XPU_ENABLE_ROPE_KERNEL")) {
    cos_buffer = reinterpret_cast<const float*>(cos_buffer_->addr_);
  }
  if (param.sin && GetBoolFromEnv("XPU_ENABLE_ROPE_KERNEL")) {
    sin_buffer = reinterpret_cast<const float*>(sin_buffer_->addr_);
  }

  xdnn::VectorParam<int> query_lod;
  if (param.SeqLod && param.SeqLod->data<int>()) {
    // vsl
    query_lod = {param.SeqLod->data<int>(),
                 static_cast<int>(param.SeqLod->numel()),
                 nullptr};
    int max_pad_seqlen = slice_idx == -1 ? param.PadSeqLen->data<int>()[0] : -1;
    xdnn::QKVAttnParam qkv_attn_param(query_lod, /* lod */
                                      param.head_num,
                                      param.size_per_head,
                                      qkv_act,
                                      slice_idx,
                                      param.enable_qkv_fusion /* qkv fusion */,
                                      max_pad_seqlen,
                                      param.hidden_dim,
                                      param.norm_before, /*is_pre_norm*/
                                      param.per_channel);
    qkv_attn_param.quant_type_.assign(quant_types_.begin(), quant_types_.end());

    if (std::is_same<TGEMM, int8_t>::value) {
      CHECK_GT(fc_input_max_.size(), 0);
    }
    int r = xdnn::transformer_encoder<T, TW, TGEMM>(
        ctx.GetRawContext(),
        in,
        *(XPUMultiEncoderCompute::get_weight<TW>()),
        out,
        fc_input_max_,
        fc_weight_max_,
        arg_fc_bias_,
        arg_ln_scale_,
        arg_ln_bias_,
        qkv_attn_param,
        nullptr, cos_buffer, sin_buffer);
    CHECK_EQ(r, 0);
  } else {
    // no vsl
    int batch = static_cast<int>(param.input->dims()[0]);
    int max_seqlen = static_cast<int>(param.input->dims()[1]);
    std::vector<int64_t> mask_shape = param.mask->dims().Vectorize();
    std::vector<int> encoder_mask_shape =
        std::vector<int>(mask_shape.begin(), mask_shape.end());

    xdnn::QKVAttnParam qkv_attn_param(batch,
                                      max_seqlen,
                                      param.head_num,
                                      param.size_per_head,
                                      encoder_mask_shape,
                                      qkv_act,
                                      slice_idx,
                                      param.enable_qkv_fusion,
                                      param.hidden_dim,
                                      param.norm_before);
    qkv_attn_param.quant_type_.assign(quant_types_.begin(), quant_types_.end());
    int r = xdnn::transformer_encoder<T, TW, TGEMM>(
        ctx.GetRawContext(),
        in,
        *(XPUMultiEncoderCompute::get_weight<TW>()),
        out,
        fc_input_max_,
        fc_weight_max_,
        arg_fc_bias_,
        arg_ln_scale_,
        arg_ln_bias_,
        qkv_attn_param,
        param.mask->data<float>(), cos_buffer, sin_buffer);
    CHECK_EQ(r, 0);
  }
}

void XPUMultiEncoderCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const float* in = param.input->data<float>();
  float* out = param.output->mutable_data<float>(TARGET(kXPU));
  if (ctx.GetRawContext()->dev().type() == xdnn::kXPU1) {
    if (param.precision == "int8") {
      run_encoder<float, int8_t, int8_t>(in, out);
    } else if (param.precision == "int16") {
      run_encoder<float, int16_t, int16_t>(in, out);
    } else if (param.precision == "int31") {
      run_encoder<float, float, int>(in, out);
    } else {
      CHECK(false);
    }
  } else {
    cast_in_guard_->Reserve(param.input->numel() * sizeof(float));
    cast_out_guard_->Reserve(param.output->numel() * sizeof(float));
    if (param.precision == "int8") {
      int r = xdnn::cast_v2<float, float16>(
          ctx.GetRawContext(),
          in,
          reinterpret_cast<float16*>(cast_in_guard_->addr_),
          param.input->numel());
      CHECK_EQ(r, 0);
      run_encoder<float16, int8_t, int8_t>(
          reinterpret_cast<const float16*>(cast_in_guard_->addr_),
          reinterpret_cast<float16*>(cast_out_guard_->addr_));
      r = xdnn::cast_v2<float16, float>(
          ctx.GetRawContext(),
          reinterpret_cast<float16*>(cast_out_guard_->addr_),
          out,
          param.output->numel());
      CHECK_EQ(r, 0);
    } else if (param.precision == "int16") {
      int r = xdnn::cast_v2<float, float16>(
          ctx.GetRawContext(),
          in,
          reinterpret_cast<float16*>(cast_in_guard_->addr_),
          param.input->numel());
      CHECK_EQ(r, 0);

      if (local_quant_) {
        run_encoder<float16, float16, float>(
            reinterpret_cast<const float16*>(cast_in_guard_->addr_),
            reinterpret_cast<float16*>(cast_out_guard_->addr_));
      } else {
        run_encoder<float16, int16_t, int16_t>(
            reinterpret_cast<const float16*>(cast_in_guard_->addr_),
            reinterpret_cast<float16*>(cast_out_guard_->addr_));
      }

      r = xdnn::cast_v2<float16, float>(
          ctx.GetRawContext(),
          reinterpret_cast<float16*>(cast_out_guard_->addr_),
          out,
          param.output->numel());
      CHECK_EQ(r, 0);
    } else if (param.precision == "int31") {
      if (local_quant_) {
        run_encoder<float, float, float>(in, out);
      } else {
        run_encoder<float, float, int>(in, out);
      }
    } else {
      CHECK(false);
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__multi_encoder,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMultiEncoderCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SeqLod",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("PadSeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Cos", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Sin", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
