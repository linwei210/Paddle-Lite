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

#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPURopeFuser : public FuseBase {
 public:
  // explicit XPURopeFuser() {}

  void BuildPattern() override {
    // OP
    auto* split = OpNode("split", "split")->AsIntermediate();
    auto* shape = OpNode("shape", "shape")->AsIntermediate();
    auto* scale = OpNode("scale", "scale")->AsIntermediate();
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* slice0 = OpNode("slice0", "slice")->AsIntermediate();
    auto* slice1 = OpNode("slice1", "slice")->AsIntermediate();
    auto* slice2 = OpNode("slice2", "slice")->AsIntermediate();
    auto* ew_mul0 = OpNode("elementwise_mul0", "elementwise_mul")->AsIntermediate();
    auto* ew_mul1 = OpNode("elementwise_mul1", "elementwise_mul")->AsIntermediate();
    auto* ew_add = OpNode("elementwise_add", "elementwise_add")->AsIntermediate();

    // Tensor
    auto* input = VarNode("T")->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input("elementwise_mul", "X")
                              ->assert_is_op_input("split", "X")
                              ->assert_is_op_input("shape", "Input")->AsInput();
    auto* split_out0 = VarNode("split_out0")->assert_is_op_nth_output("split", "Out", 0)
                                            ->assert_is_op_nth_input("concat", "X", 1)->AsIntermediate();
    auto* split_out1 = VarNode("split_out1")->assert_is_op_nth_output("split", "Out", 1)
                                            ->assert_is_op_input("scale", "X")->AsIntermediate();
    auto* scale_out = VarNode("scale_out")->assert_is_op_output("scale", "Out")
                                          ->assert_is_op_nth_input("concat", "X", 0)->AsIntermediate();
    auto* concat_out = VarNode("concat_out")->assert_is_op_output("concat", "Out")
                                          ->assert_is_op_input("elementwise_mul", "X")->AsIntermediate();

    auto* shape_out = VarNode("shape_out")->assert_is_op_output("shape", "Out")
                                      ->assert_is_op_input("slice", "Input")->AsIntermediate();
    auto* slice0_out = VarNode("slice0_out")->assert_is_op_output("slice", "Out")
                                      ->assert_is_op_input("slice", "EndsTensorList")
                                      ->assert_is_op_input("slice", "EndsTensorList")
                                      ->AsIntermediate();
    auto* cos = VarNode("cos")->assert_is_op_input("slice", "Input")->AsInput();
    auto* sin = VarNode("sin")->assert_is_op_input("slice", "Input")->AsInput();
    auto* slice1_out = VarNode("slice1_out")->assert_is_op_output("slice", "Out")
                                      ->assert_is_op_input("elementwise_mul", "Y")
                                      ->AsIntermediate();
    auto* slice2_out = VarNode("slice2_out")->assert_is_op_output("slice", "Out")
                                  ->assert_is_op_input("elementwise_mul", "Y")
                                  ->AsIntermediate();

    auto* ew_mul0_out = VarNode("ew_mul0_out")->assert_is_op_output("elementwise_mul", "Out")
                                      ->assert_is_op_input("elementwise_add", "X")->AsIntermediate();
    auto* ew_mul1_out = VarNode("ew_mul1_out")->assert_is_op_output("elementwise_mul", "Out")
                                    ->assert_is_op_input("elementwise_add", "Y")->AsIntermediate();
    auto* ew_add_out = VarNode("ew_add_out")->assert_is_op_output("elementwise_add", "Out")->AsOutput();

    *input >> *shape >> *shape_out >> *slice0 >> *slice0_out >> *slice1 >> *slice1_out >> *ew_mul0 >> *ew_mul0_out >> *ew_add >> *ew_add_out;
    *cos >> *slice1;
    *input >> *ew_mul0;

    *slice0_out >> *slice2 >> *slice2_out >> *ew_mul1 >> *ew_mul1_out >> *ew_add;
    *sin >> *slice2;
    *input >> *split >> *split_out0 >> *concat >> *concat_out >> *ew_mul1;
    *split >> *split_out1 >> *scale >> *scale_out >> *concat;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::cout << "Begin to InsertNewNode" << std::endl;
    cpp::OpDesc op_desc = *matched.at("split")->stmt()->op_info();
    auto split = matched.at("split")->stmt()->op();
    auto* scope = split->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__rope");
    op_desc.SetInput("Input", {matched.at("T")->arg()->name});
    op_desc.SetInput("Cos", {matched.at("cos")->arg()->name});
    op_desc.SetInput("Sin", {matched.at("sin")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("ew_add_out")->arg()->name});

    auto rope_op = LiteOpRegistry::Global().Create("__xpu__rope");
    auto& valid_places = split->valid_places();
    rope_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(rope_op, valid_places);

    IR_NODE_LINK_TO(matched.at("T"), new_op_node);
    IR_NODE_LINK_TO(matched.at("cos"), new_op_node);
    IR_NODE_LINK_TO(matched.at("sin"), new_op_node);
    IR_OP_VAR_LINK(new_op_node, matched.at("ew_add_out"));
    std::cout << "InsertNewNode DONE" << std::endl;
  }
};

}  // namespace fusion

class XPURopeFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    if (!GetBoolFromEnv("XPU_ENABLE_ROPE")) return;
    fusion::XPURopeFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__rope_fuse_pass, paddle::lite::mir::XPURopeFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__rope");
