/* Copyright 2026 The xLLM Authors.

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

#include <exception>
#include <nlohmann/json.hpp>
#include <string>

namespace xllm_service {
namespace {

bool omit_empty_tool_calls(std::string* response_json, std::string* error) {
  if (response_json->find("\"tool_calls\":[]") == std::string::npos) {
    return true;
  }

  try {
    nlohmann::ordered_json response =
        nlohmann::ordered_json::parse(*response_json);
    auto choices = response.find("choices");
    if (choices == response.end() || !choices->is_array()) {
      return true;
    }

    for (auto& choice : *choices) {
      if (!choice.is_object()) {
        continue;
      }
      for (const char* message_key : {"message", "delta"}) {
        auto message = choice.find(message_key);
        if (message == choice.end() || !message->is_object()) {
          continue;
        }
        auto tool_calls = message->find("tool_calls");
        if (tool_calls != message->end() && tool_calls->is_array() &&
            tool_calls->empty()) {
          message->erase(tool_calls);
        }
      }
    }

    *response_json = response.dump();
    return true;
  } catch (const nlohmann::json::exception& e) {
    *error = "Failed to remove empty tool calls: " + std::string(e.what());
    return false;
  } catch (const std::exception& e) {
    *error = "Failed to remove empty tool calls: " + std::string(e.what());
    return false;
  }
}

}  // namespace

bool ChatCallData::write_and_finish(::xllm::proto::ChatResponse& response) {
  std::string response_json;
  std::string error;
  if (!serialize_response(response, &response_json, &error) ||
      !omit_empty_tool_calls(&response_json, &error)) {
    return finish_with_error(error);
  }
  return Base::write_and_finish(response_json);
}

bool ChatCallData::write(::xllm::proto::ChatResponse& response) {
  std::string response_json;
  std::string error;
  if (!serialize_response(response, &response_json, &error) ||
      !omit_empty_tool_calls(&response_json, &error)) {
    LOG(ERROR) << error;
    return false;
  }
  return Base::write("data: " + response_json + "\n\n");
}

}  // namespace xllm_service
