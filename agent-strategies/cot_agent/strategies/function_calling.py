import json
import time
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Optional, cast

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import (
    LLMModelConfig,
    LLMResult,
    LLMResultChunk,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage, ToolParameter, ToolProviderType
from dify_plugin.entities.model.message import PromptMessageTool
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentStrategy,
    ToolEntity,
    ToolInvokeMeta,
)
from pydantic import BaseModel
import logging
from dify_plugin.config.logger_format import plugin_logger_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)

class LogMetadata:
    """Metadata keys for logging"""
    STARTED_AT = "started_at"
    PROVIDER = "provider"
    FINISHED_AT = "finished_at"
    ELAPSED_TIME = "elapsed_time"
    TOTAL_PRICE = "total_price"
    CURRENCY = "currency"
    TOTAL_TOKENS = "total_tokens"

class ExecutionMetadata(BaseModel):
    """Execution metadata with default values"""
    total_price: float = 0.0
    currency: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    prompt_unit_price: float = 0.0
    prompt_price_unit: float = 0.0
    prompt_price: float = 0.0
    completion_tokens: int = 0
    completion_unit_price: float = 0.0
    completion_price_unit: float = 0.0
    completion_price: float = 0.0
    latency: float = 0.0
    
    @classmethod
    def from_llm_usage(cls, usage: Optional[LLMUsage]) -> "ExecutionMetadata":
        """Create ExecutionMetadata from LLMUsage, handling None case"""
        if usage is None:
            return cls()
        
        return cls(
            total_price=float(usage.total_price),
            currency=usage.currency,
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            prompt_unit_price=float(usage.prompt_unit_price),
            prompt_price_unit=float(usage.prompt_price_unit),
            prompt_price=float(usage.prompt_price),
            completion_tokens=usage.completion_tokens,
            completion_unit_price=float(usage.completion_unit_price),
            completion_price_unit=float(usage.completion_price_unit),
            completion_price=float(usage.completion_price),
            latency=usage.latency
        )

class ContextItem(BaseModel):
    content: str
    title: str
    metadata: dict[str, Any]


class FunctionCallingParams(BaseModel):
    query: str
    instruction: str | None
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    maximum_iterations: int = 3
    context: list[ContextItem] | None = None


class FunctionCallingAgentStrategy(AgentStrategy):
    query: str = ""
    instruction: str | None = ""

    # JSON Schema for FILE type parameters
    FILE_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "related_id": {
                "type": "string",
                "description": "File related_id, must be UUID format.",
            },
            "transfer_method": {
                "type": "string",
                "description": "File transfer method.",
                "enum": [
                    "remote_url",
                    "local_file",
                    "tool_file",
                    "datasource_file",
                ],
            },
        },
        "required": ["related_id", "transfer_method"],
    }

    @property
    def _user_prompt_message(self) -> UserPromptMessage:
        return UserPromptMessage(content=self.query)

    @property
    def _system_prompt_message(self) -> SystemPromptMessage:
        return SystemPromptMessage(content=self.instruction)

    def _invoke(
        self, parameters: dict[str, Any]
    ) -> Generator[AgentInvokeMessage, None, None]:
        """
        Run FunctionCall agent application
        """
        fc_params = FunctionCallingParams(**parameters)

        # init prompt messages
        query = fc_params.query
        self.query = query
        self.instruction = fc_params.instruction
        history_prompt_messages = fc_params.model.history_prompt_messages
        history_prompt_messages.insert(0, self._system_prompt_message)
        history_prompt_messages.append(self._user_prompt_message)

        # convert tool messages
        tools = fc_params.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}
        prompt_messages_tools = self._init_prompt_tools(tools)

        # init model parameters
        stream = (
            ModelFeature.STREAM_TOOL_CALL in fc_params.model.entity.features
            if fc_params.model.entity and fc_params.model.entity.features
            else False
        )
        model = fc_params.model
        stop = (
            fc_params.model.completion_params.get("stop", [])
            if fc_params.model.completion_params
            else []
        )

        # init function calling state
        iteration_step = 1
        max_iteration_steps = fc_params.maximum_iterations
        current_thoughts: list[PromptMessage] = []
        function_call_state = True  # continue to run until there is not any tool call
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""

        while function_call_state and iteration_step <= max_iteration_steps:
            # start a new round
            function_call_state = False
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                },
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            # If max_iteration_steps=1, need to execute tool calls
            if iteration_step == max_iteration_steps and max_iteration_steps > 1:
                # the last iteration, remove all tools
                prompt_messages_tools = []

            # recalc llm max tokens
            prompt_messages = self._organize_prompt_messages(
                history_prompt_messages=history_prompt_messages,
                current_thoughts=current_thoughts,
            )
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(
                    model.entity, prompt_messages, model.completion_params
                )
            # invoke model
            model_started_at = time.perf_counter()
            model_log = self.create_log_message(
                label=f"{model.model} Thought",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                },
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log
            model_config = LLMModelConfig(**model.model_dump(mode="json"))
            chunks: Generator[LLMResultChunk, None, None] | LLMResult = (
                self.session.model.llm.invoke(
                    model_config=model_config,
                    prompt_messages=prompt_messages,
                    stop=stop,
                    stream=stream,
                    tools=prompt_messages_tools,
                )
            )

            tool_calls: list[tuple[str, str, dict[str, Any]]] = []

            # save full response
            response = ""

            # save tool call names and inputs
            tool_call_names = ""

            current_llm_usage = None

            if isinstance(chunks, Generator):
                for chunk in chunks:
                    # check if there is any tool call
                    if self.check_tool_calls(chunk):
                        function_call_state = True
                        tool_calls.extend(self.extract_tool_calls(chunk) or [])
                        tool_call_names = ";".join(
                            [tool_call[1] for tool_call in tool_calls]
                        )

                    if chunk.delta.message and chunk.delta.message.content:
                        if isinstance(chunk.delta.message.content, list):
                            for content in chunk.delta.message.content:
                                response += content.data
                                if (
                                    not function_call_state
                                    or iteration_step == max_iteration_steps
                                ):
                                    yield self.create_text_message(content.data)
                        else:
                            response += str(chunk.delta.message.content)
                            if (
                                not function_call_state
                                or iteration_step == max_iteration_steps
                            ):
                                yield self.create_text_message(
                                    str(chunk.delta.message.content)
                                )

                    if chunk.delta.usage:
                        self.increase_usage(llm_usage, chunk.delta.usage)
                        current_llm_usage = chunk.delta.usage

            else:
                result = chunks
                result = cast(LLMResult, result)
                # check if there is any tool call
                if self.check_blocking_tool_calls(result):
                    function_call_state = True
                    tool_calls.extend(self.extract_blocking_tool_calls(result) or [])
                    tool_call_names = ";".join(
                        [tool_call[1] for tool_call in tool_calls]
                    )

                if result.usage:
                    self.increase_usage(llm_usage, result.usage)
                    current_llm_usage = result.usage

                if result.message and result.message.content:
                    if isinstance(result.message.content, list):
                        for content in result.message.content:
                            response += content.data
                    else:
                        response += str(result.message.content)

                if not result.message.content:
                    result.message.content = ""
                if isinstance(result.message.content, str):
                    yield self.create_text_message(result.message.content)
                elif isinstance(result.message.content, list):
                    for content in result.message.content:
                        yield self.create_text_message(content.data)

            yield self.finish_log_message(
                log=model_log,
                data={
                    "output": response,
                    "tool_name": tool_call_names,
                    "tool_input": [
                        {"name": tool_call[1], "args": tool_call[2]}
                        for tool_call in tool_calls
                    ],
                },
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                    LogMetadata.TOTAL_PRICE: current_llm_usage.total_price
                    if current_llm_usage
                    else 0,
                    LogMetadata.CURRENCY: current_llm_usage.currency
                    if current_llm_usage
                    else "",
                    LogMetadata.TOTAL_TOKENS: current_llm_usage.total_tokens
                    if current_llm_usage
                    else 0,
                },
            )

            # If there are tool calls, merge all tool calls into a single assistant message
            if tool_calls:
                tool_call_objects = [
                    AssistantPromptMessage.ToolCall(
                        id=tool_call_id,
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=tool_call_name,
                            arguments=json.dumps(
                                tool_call_args, ensure_ascii=False
                            ),
                        ),
                    )
                    for tool_call_id, tool_call_name, tool_call_args in tool_calls
                ]
                assistant_message = AssistantPromptMessage(
                    content=response,  # Preserve LLM returned content, even if empty
                    tool_calls=tool_call_objects
                )
                current_thoughts.append(assistant_message)
            elif response.strip():
                # If no tool calls but has response, add a regular assistant message
                assistant_message = AssistantPromptMessage(
                    content=response, tool_calls=[]
                )
                current_thoughts.append(assistant_message)

            final_answer += response + "\n"

            # call tools
            tool_responses = []
            for tool_call_id, tool_call_name, tool_call_args in tool_calls:
                tool_instance = tool_instances[tool_call_name]
                tool_call_started_at = time.perf_counter()
                tool_call_log = self.create_log_message(
                    label=f"CALL {tool_call_name}",
                    data={},
                    metadata={
                        LogMetadata.STARTED_AT: time.perf_counter(),
                        LogMetadata.PROVIDER: tool_instance.identity.provider,
                    },
                    parent=round_log,
                    status=ToolInvokeMessage.LogMessage.LogStatus.START,
                )
                yield tool_call_log
                if not tool_instance:
                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_response": f"there is not a tool named {tool_call_name}",
                        "meta": ToolInvokeMeta.error_instance(
                            f"there is not a tool named {tool_call_name}"
                        ).to_dict(),
                    }
                else:
                    # invoke tool
                    try:
                        tool_invoke_responses = self.session.tool.invoke(
                            provider_type=ToolProviderType(tool_instance.provider_type),
                            provider=tool_instance.identity.provider,
                            tool_name=tool_instance.identity.name,
                            parameters={
                                **tool_instance.runtime_parameters,
                                **tool_call_args,
                            },
                        )
                        tool_result = ""
                        for tool_invoke_response in tool_invoke_responses:
                            logger.info(f"!!!!!!!!!!!!!!!!!!!!Serializing tool response: {tool_invoke_response}")
                            tool_result += self._serialize_tool_response(tool_invoke_response)
                            if tool_invoke_response.type in {
                                ToolInvokeMessage.MessageType.IMAGE_LINK,
                                ToolInvokeMessage.MessageType.IMAGE,
                                ToolInvokeMessage.MessageType.BLOB,
                                ToolInvokeMessage.MessageType.FILE,
                                "binary_link"
                            }:
                                yield tool_invoke_response
                    except Exception as e:
                        tool_result = f"tool invoke error: {e!s}"
                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_call_input": {
                            **tool_instance.runtime_parameters,
                            **tool_call_args,
                        },
                        "tool_response": tool_result,
                    }

                yield self.finish_log_message(
                    log=tool_call_log,
                    data={
                        "output": tool_response,
                    },
                    metadata={
                        LogMetadata.STARTED_AT: tool_call_started_at,
                        LogMetadata.PROVIDER: tool_instance.identity.provider,
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter()
                        - tool_call_started_at,
                    },
                )
                tool_responses.append(tool_response)
                if tool_response["tool_response"] is not None:
                    current_thoughts.append(
                        ToolPromptMessage(
                            content=str(tool_response["tool_response"]),
                            tool_call_id=tool_call_id,
                            name=tool_call_name,
                        )
                    )
            # After handling all tool calls, insert a blank line so the next assistant thought
            # appears on a new line in the user interface.
            if tool_calls:
                yield self.create_text_message("\n")

            # update prompt tool
            for prompt_tool in prompt_messages_tools:
                self.update_prompt_message_tool(
                    tool_instances[prompt_tool.name], prompt_tool
                )
            yield self.finish_log_message(
                log=round_log,
                data={
                    "output": {
                        "llm_response": response,
                        "tool_responses": tool_responses,
                    },
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_PRICE: current_llm_usage.total_price
                    if current_llm_usage
                    else 0,
                    LogMetadata.CURRENCY: current_llm_usage.currency
                    if current_llm_usage
                    else "",
                    LogMetadata.TOTAL_TOKENS: current_llm_usage.total_tokens
                    if current_llm_usage
                    else 0,
                },
            )
            # If max_iteration_steps=1, need to return tool responses
            if tool_responses and max_iteration_steps == 1:
                for resp in tool_responses:
                    yield self.create_text_message(str(resp["tool_response"]))
            iteration_step += 1

        # If context is a list of dict, create retriever resource message
        if isinstance(fc_params.context, list):
            yield self.create_retriever_resource_message(
                retriever_resources=[
                    ToolInvokeMessage.RetrieverResourceMessage.RetrieverResource(
                        content=ctx.content,
                        position=ctx.metadata.get("position"),
                        dataset_id=ctx.metadata.get("dataset_id"),
                        dataset_name=ctx.metadata.get("dataset_name"),
                        document_id=ctx.metadata.get("document_id"),
                        document_name=ctx.metadata.get("document_name"),
                        data_source_type=ctx.metadata.get("document_data_source_type"),
                        segment_id=ctx.metadata.get("segment_id"),
                        retriever_from=ctx.metadata.get("retriever_from"),
                        score=ctx.metadata.get("score"),
                        hit_count=ctx.metadata.get("segment_hit_count"),
                        word_count=ctx.metadata.get("segment_word_count"),
                        segment_position=ctx.metadata.get("segment_position"),
                        index_node_hash=ctx.metadata.get("segment_index_node_hash"),
                        page=ctx.metadata.get("page"),
                        doc_metadata=ctx.metadata.get("doc_metadata"),
                    )
                    for ctx in fc_params.context
                ],
                context="",
            )

        metadata = ExecutionMetadata.from_llm_usage(llm_usage["usage"])
        yield self.create_json_message(
            {
                "execution_metadata": metadata.model_dump()
            }
        )

    def check_tool_calls(self, llm_result_chunk: LLMResultChunk) -> bool:
        """
        Check if there is any tool call in llm result chunk
        """
        return bool(llm_result_chunk.delta.message.tool_calls)

    def check_blocking_tool_calls(self, llm_result: LLMResult) -> bool:
        """
        Check if there is any blocking tool call in llm result
        """
        return bool(llm_result.message.tool_calls)

    def extract_tool_calls(
        self, llm_result_chunk: LLMResultChunk
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Extract tool calls from llm result chunk

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: [(tool_call_id, tool_call_name, tool_call_args)]
        """
        tool_calls = []
        for prompt_message in llm_result_chunk.delta.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)

            tool_calls.append(
                (
                    prompt_message.id,
                    prompt_message.function.name,
                    args,
                )
            )

        return tool_calls

    def extract_blocking_tool_calls(
        self, llm_result: LLMResult
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Extract blocking tool calls from llm result

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: [(tool_call_id, tool_call_name, tool_call_args)]
        """
        tool_calls = []
        for prompt_message in llm_result.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)

            tool_calls.append(
                (
                    prompt_message.id,
                    prompt_message.function.name,
                    args,
                )
            )

        return tool_calls

    def _init_system_message(
        self, prompt_template: str, prompt_messages: list[PromptMessage]
    ) -> list[PromptMessage]:
        """
        Initialize system message
        """
        if not prompt_messages and prompt_template:
            return [
                SystemPromptMessage(content=prompt_template),
            ]

        if (
            prompt_messages
            and not isinstance(prompt_messages[0], SystemPromptMessage)
            and prompt_template
        ):
            prompt_messages.insert(0, SystemPromptMessage(content=prompt_template))

        return prompt_messages or []

    def _clear_user_prompt_image_messages(
        self, prompt_messages: list[PromptMessage]
    ) -> list[PromptMessage]:
        """
        As for now, gpt supports both fc and vision at the first iteration.
        We need to remove the image messages from the prompt messages at the first iteration.
        """
        prompt_messages = deepcopy(prompt_messages)

        for prompt_message in prompt_messages:
            if isinstance(prompt_message, UserPromptMessage) and isinstance(
                prompt_message.content, list
            ):
                prompt_message.content = "\n".join(
                    [
                        content.data
                        if content.type == PromptMessageContentType.TEXT
                        else "[image]"
                        if content.type == PromptMessageContentType.IMAGE
                        else "[file]"
                        for content in prompt_message.content
                    ]
                )

        return prompt_messages

    def _organize_prompt_messages(
        self,
        current_thoughts: list[PromptMessage],
        history_prompt_messages: list[PromptMessage],
    ) -> list[PromptMessage]:
        prompt_messages = [
            *history_prompt_messages,
            *current_thoughts,
        ]
        if len(current_thoughts) != 0:
            # clear messages after the first iteration
            prompt_messages = self._clear_user_prompt_image_messages(prompt_messages)
        return prompt_messages

    def _convert_tool_to_prompt_message_tool(self, tool: ToolEntity) -> PromptMessageTool:
        """
        convert tool to prompt message tool
        """
        message_tool = PromptMessageTool(
            name=tool.identity.name,
            description=tool.description.llm if tool.description else "",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

        parameters = tool.parameters
        for parameter in parameters:
            if parameter.form != ToolParameter.ToolParameterForm.LLM:
                continue

            parameter_type = parameter.type
            if parameter.type in {
                ToolParameter.ToolParameterType.SELECT,
                ToolParameter.ToolParameterType.SECRET_INPUT,
            }:
                parameter_type = ToolParameter.ToolParameterType.STRING
            enum = []
            if parameter.type == ToolParameter.ToolParameterType.SELECT:
                enum = [option.value for option in parameter.options] if parameter.options else []

            message_tool.parameters["properties"][parameter.name] = (
                {
                    "type": parameter_type,
                    "description": parameter.llm_description or "",
                }
                if parameter.input_schema is None
                else parameter.input_schema
            )

            # Handle FILE type parameters with special schema
            if parameter.type == ToolParameter.ToolParameterType.FILE:
                logger.info(f"Handling FILE parameter: {parameter.name} for tool: {tool.identity.name}")
                message_tool.parameters["properties"][parameter.name] = self.FILE_JSON_SCHEMA
            elif parameter.type == ToolParameter.ToolParameterType.FILES:
                logger.info(f"Handling FILES parameter: {parameter.name} for tool: {tool.identity.name}")
                message_tool.parameters["properties"][parameter.name] = {
                    "type": "array",
                    "items": self.FILE_JSON_SCHEMA,
                    "description": parameter.llm_description or "",
                }

            if len(enum) > 0:
                message_tool.parameters["properties"][parameter.name]["enum"] = enum

            if parameter.required:
                message_tool.parameters["required"].append(parameter.name)

        return message_tool

    def update_prompt_message_tool(self, tool: ToolEntity, prompt_tool: PromptMessageTool) -> PromptMessageTool:
        """
        update prompt message tool
        """
        # try to get tool runtime parameters
        tool_runtime_parameters = tool.parameters

        for parameter in tool_runtime_parameters:
            if parameter.form != ToolParameter.ToolParameterForm.LLM:
                continue

            parameter_type = parameter.type
            if parameter.type in {
                ToolParameter.ToolParameterType.SELECT,
                ToolParameter.ToolParameterType.SECRET_INPUT,
            }:
                parameter_type = ToolParameter.ToolParameterType.STRING
            enum = []
            if parameter.type == ToolParameter.ToolParameterType.SELECT:
                enum = [option.value for option in parameter.options] if parameter.options else []

            prompt_tool.parameters["properties"][parameter.name] = (
                {
                    "type": parameter_type,
                    "description": parameter.llm_description or "",
                }
                if parameter.input_schema is None
                else parameter.input_schema
            )

            # Handle FILE type parameters with special schema
            if parameter.type == ToolParameter.ToolParameterType.FILE:
                logger.info(f"Updating FILE parameter: {parameter.name} for tool: {tool.identity.name}")
                prompt_tool.parameters["properties"][parameter.name] = self.FILE_JSON_SCHEMA
            elif parameter.type == ToolParameter.ToolParameterType.FILES:
                logger.info(f"Updating FILES parameter: {parameter.name} for tool: {tool.identity.name}")
                prompt_tool.parameters["properties"][parameter.name] = {
                    "type": "array",
                    "items": self.FILE_JSON_SCHEMA,
                    "description": parameter.llm_description or "",
                }

            if len(enum) > 0:
                prompt_tool.parameters["properties"][parameter.name]["enum"] = enum

            if parameter.required and parameter.name not in prompt_tool.parameters["required"]:
                prompt_tool.parameters["required"].append(parameter.name)

        return prompt_tool

    def _serialize_tool_response(self, response: ToolInvokeMessage) -> str:
        """
        Serialize tool response for LLM
        """
        logger.info(f"Serializing tool response: {response}")
        result = ""
        if response.type == ToolInvokeMessage.MessageType.TEXT:
            result += cast(ToolInvokeMessage.TextMessage, response.message).text
        elif response.type == ToolInvokeMessage.MessageType.LINK:
            result += (
                f"result link: {cast(ToolInvokeMessage.TextMessage, response.message).text}."
                + " please tell user to check it."
            )
        elif response.type in {
            ToolInvokeMessage.MessageType.IMAGE_LINK,
            ToolInvokeMessage.MessageType.IMAGE,
            ToolInvokeMessage.MessageType.BLOB,
            "binary_link"
            } :
            if isinstance(response.message, ToolInvokeMessage.TextMessage):
                message_text = response.message.text
            else:
                message_text = str(response.message)
            try:
                tool_file_id = message_text.split("/")[-1].split(".")[0]
                serialized = json.dumps({
                        "related_id": tool_file_id,
                        "transfer_method": "tool_file",
                        }, ensure_ascii=False)
                result += serialized
            except Exception as e:
                logger.info(f"Failed to parse tool_file_id from {message_text}: {e}")
                result += message_text
        elif response.type == ToolInvokeMessage.MessageType.JSON:
            text = json.dumps(
                cast(
                    ToolInvokeMessage.JsonMessage, response.message
                ).json_object,
                ensure_ascii=False,
            )
            result += f"tool response: {text}."
        elif response.type == ToolInvokeMessage.MessageType.FILE:
            logger.info("Serializing FILE response")
            if response.meta and "file" in response.meta:
                file_obj = response.meta["file"]
                if hasattr(file_obj, "related_id"):
                    serialized = json.dumps({
                        "related_id": file_obj.related_id,
                        "filename": getattr(file_obj, "filename", ""),
                        "extension": getattr(file_obj, "extension", ""),
                        "mime_type": getattr(file_obj, "mimetype", ""),
                        "transfer_method": str(getattr(file_obj, "transfer_method", "tool_file")),
                    }, ensure_ascii=False)
                    logger.info(f"Serialized FILE: {serialized}")
                    result += serialized
                else:
                    logger.info("FILE object has no related_id")
                    result += "File received."
            else:
                logger.info("FILE message has no file in meta")
                result += "File received."
        else:
            result += f"tool response: {response.message!r}."
        
        return result

