/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "common/call_data.h"

#include <brpc/controller.h>
#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace xllm_service {
namespace {

class TestClosure : public google::protobuf::Closure {
 public:
  void Run() override { ran = true; }
  bool ran = false;
};

TEST(CallDataTest, ChatRequestParsesJsonObjectResponseFormat) {
  xllm::proto::ChatRequest request;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;

  const auto status = google::protobuf::util::JsonStringToMessage(
      R"({"model":"test","messages":[{"role":"user","content":"ping"}],
          "response_format":{"type":"json_object"}})",
      &request,
      options);

  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(request.has_response_format());
  EXPECT_EQ(request.response_format().type(), "json_object");
}

TEST(CallDataTest, NonStreamWriteAndFinishTracesAttachment) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;
  std::vector<std::string> trace_chunks;

  {
    ChatCallData call_data(
        &controller,
        /*stream=*/false,
        &done,
        &request,
        &response,
        [&](const std::string& chunk) { trace_chunks.push_back(chunk); });

    ASSERT_FALSE(done.ran);
    ASSERT_TRUE(call_data.write_and_finish("{\"id\":\"chatcmpl-test\"}"));

    ASSERT_EQ(trace_chunks.size(), 1);
    EXPECT_EQ(trace_chunks[0], "{\"id\":\"chatcmpl-test\"}");
    EXPECT_EQ(controller.response_attachment().to_string(),
              "{\"id\":\"chatcmpl-test\"}");
  }

  EXPECT_TRUE(done.ran);
}

TEST(CallDataTest, NonStreamFinishWithErrorMarksControllerFailed) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;

  {
    ChatCallData call_data(&controller,
                           /*stream=*/false,
                           &done,
                           &request,
                           &response);

    ASSERT_TRUE(call_data.finish_with_error("encode failed"));
    EXPECT_TRUE(controller.Failed());
    EXPECT_EQ(controller.ErrorText(), "encode failed");
  }

  EXPECT_TRUE(done.ran);
}

TEST(CallDataTest, ChatResponseOmitsEmptyToolCalls) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;

  auto* choice = response.add_choices();
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  message->set_content("pong");

  {
    ChatCallData call_data(&controller,
                           /*stream=*/false,
                           &done,
                           &request,
                           &response);
    ASSERT_TRUE(call_data.write_and_finish(response));
  }

  EXPECT_EQ(controller.response_attachment().to_string().find("\"tool_calls\""),
            std::string::npos);
  EXPECT_TRUE(done.ran);
}

TEST(CallDataTest, ChatResponseKeepsNonEmptyToolCalls) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;

  auto* choice = response.add_choices();
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  auto* tool_call = message->add_tool_calls();
  tool_call->set_id("call_test");
  tool_call->set_type("function");
  tool_call->mutable_function()->set_name("get_weather");
  tool_call->mutable_function()->set_arguments("{\"city\":\"Beijing\"}");

  {
    ChatCallData call_data(&controller,
                           /*stream=*/false,
                           &done,
                           &request,
                           &response);
    ASSERT_TRUE(call_data.write_and_finish(response));
  }

  const std::string response_json =
      controller.response_attachment().to_string();
  EXPECT_NE(response_json.find("\"tool_calls\""), std::string::npos);
  EXPECT_NE(response_json.find("\"name\":\"get_weather\""), std::string::npos);
  EXPECT_TRUE(done.ran);
}

TEST(CallDataTest, ChatResponseKeepsEmptyTopLogprobs) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;

  auto* choice = response.add_choices();
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  message->set_content("pong");
  auto* logprob = choice->mutable_logprobs()->add_content();
  logprob->set_token("pong");
  logprob->set_token_id(1);

  {
    ChatCallData call_data(&controller,
                           /*stream=*/false,
                           &done,
                           &request,
                           &response);
    ASSERT_TRUE(call_data.write_and_finish(response));
  }

  const std::string response_json =
      controller.response_attachment().to_string();
  EXPECT_EQ(response_json.find("\"tool_calls\""), std::string::npos);
  EXPECT_NE(response_json.find("\"top_logprobs\":[]"), std::string::npos);
  EXPECT_TRUE(done.ran);
}

TEST(CallDataTest, CompletionResponseKeepsEmptyChoices) {
  brpc::Controller controller;
  xllm::proto::CompletionRequest request;
  xllm::proto::CompletionResponse response;
  TestClosure done;

  response.mutable_choices();

  {
    CompletionCallData call_data(&controller,
                                 /*stream=*/false,
                                 &done,
                                 &request,
                                 &response);
    ASSERT_TRUE(call_data.write_and_finish(response));
  }

  EXPECT_NE(controller.response_attachment().to_string().find("\"choices\":[]"),
            std::string::npos);
  EXPECT_TRUE(done.ran);
}

}  // namespace
}  // namespace xllm_service
