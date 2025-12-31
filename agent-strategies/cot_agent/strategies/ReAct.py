import json
import time
from collections.abc import Generator, Mapping
from typing import Any, Optional, cast

import pydantic
from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import (
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)

from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentScratchpadUnit,
    AgentStrategy,
    ToolEntity,
)
import logging
from dify_plugin.config.logger_format import plugin_logger_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)
from output_parser.cot_output_parser import ReactChunk, ReactState, CotAgentOutputParser
from prompt.template import REACT_PROMPT_TEMPLATES
from pydantic import BaseModel, Field

class LogMetadata:
    """Metadata keys for logging"""
    STARTED_AT = "started_at"
    PROVIDER = "provider"
    FINISHED_AT = "finished_at"
    ELAPSED_TIME = "elapsed_time"
    TOTAL_PRICE = "total_price"
    CURRENCY = "currency"
    TOTAL_TOKENS = "total_tokens"

ignore_observation_providers = ["wenxin"]


class ContextItem(BaseModel):
    content: str
    title: str
    metadata: dict[str, Any]


class ReActParams(BaseModel):
    query: str
    instruction: str
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    maximum_iterations: int = 3
    context: list[ContextItem] | None = None


class AgentPromptEntity(BaseModel):
    """
    Agent Prompt Entity.
    """

    first_prompt: str
    next_iteration: str


class ReActAgentStrategy(AgentStrategy):
    query: str = ""
    instruction: str = ""
    history_prompt_messages: list[PromptMessage] = Field(default_factory=list)
    prompt_messages_tools: list[ToolEntity] = Field(default_factory=list)

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
        prompt_entity = AgentPromptEntity(
            first_prompt=REACT_PROMPT_TEMPLATES["english"]["chat"]["prompt"],
            next_iteration=REACT_PROMPT_TEMPLATES["english"]["chat"][
                "agent_scratchpad"
            ],
        )
        if not prompt_entity:
            raise ValueError("Agent prompt configuration is not set")
        first_prompt = prompt_entity.first_prompt

        system_prompt = (
            first_prompt.replace("{{instruction}}", self.instruction)
            .replace(
                "{{tools}}",
                json.dumps(
                    [
                        tool.model_dump(mode="json")
                        for tool in self._prompt_messages_tools
                    ]
                ),
            )
            .replace(
                "{{tool_names}}",
                ", ".join([tool.name for tool in self._prompt_messages_tools]),
            )
        )

        return SystemPromptMessage(content=system_prompt)

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        try:
            react_params = ReActParams(**parameters)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid parameters: {e!s}") from e

        # Init parameters
        self.query = react_params.query
        self.instruction = react_params.instruction
        agent_scratchpad: list[AgentScratchpadUnit] = []
        iteration_step = 1
        max_iteration_steps = react_params.maximum_iterations
        run_agent_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""
        prompt_messages: list[PromptMessage] = []

        # Init model
        model = react_params.model
        stop = (
            react_params.model.completion_params.get("stop", [])
            if react_params.model.completion_params
            else []
        )
        if (
            "Observation" not in stop
            and model.provider not in ignore_observation_providers
        ):
            stop.append("Observation")

        # Init prompts
        self.history_prompt_messages = model.history_prompt_messages

        # convert tools into ModelRuntime Tool format
        tools = react_params.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}
        react_params.model.completion_params = (
            react_params.model.completion_params or {}
        )
        prompt_messages_tools = self._init_prompt_tools(tools)
        self._prompt_messages_tools = prompt_messages_tools

        while run_agent_state and iteration_step <= max_iteration_steps:
            # continue to run until there is not any tool call
            run_agent_state = False
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
            if iteration_step == max_iteration_steps:
                # the last iteration, remove all tools
                self._prompt_messages_tools = []

            message_file_ids: list[str] = []

            # recalc llm max tokens
            prompt_messages = self._organize_prompt_messages(
                agent_scratchpad, self.query
            )
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(
                    model.entity, prompt_messages, model.completion_params
                )
            # invoke model
            chunks = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stream=True,
                stop=stop,
            )

            usage_dict: dict[str, Optional[LLMUsage]] = {"usage": None}
            react_chunks = CotAgentOutputParser.handle_react_stream_output(
                chunks, usage_dict
            )
            scratchpad = AgentScratchpadUnit(
                agent_response="",
                thought="",
                action_str="",
                observation="",
                action=None,
            )

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

            for react_chunk in react_chunks:
                if isinstance(react_chunk, AgentScratchpadUnit.Action):
                    action = react_chunk
                    # detect action
                    assert scratchpad.agent_response is not None
                    scratchpad.agent_response += json.dumps(react_chunk.model_dump())

                    scratchpad.action_str = json.dumps(react_chunk.model_dump())
                    scratchpad.action = action
                else:
                    assert isinstance(react_chunk, ReactChunk)
                    chunk_state = react_chunk.state
                    chunk = react_chunk.content
                    yield self.create_text_message(chunk)
                    if chunk_state == ReactState.ANSWER:
                        final_answer += chunk
                    elif chunk_state == ReactState.THINKING:
                        scratchpad.agent_response = scratchpad.agent_response or ""
                        scratchpad.thought = scratchpad.thought or ""
                        scratchpad.agent_response += chunk
                        scratchpad.thought += chunk
            scratchpad.thought = (
                scratchpad.thought.strip()
                if scratchpad.thought
                else "I am thinking about how to help you"
            )
            agent_scratchpad.append(scratchpad)

            # get llm usage
            if "usage" in usage_dict:
                if usage_dict["usage"] is not None:
                    self.increase_usage(llm_usage, usage_dict["usage"])
            else:
                usage_dict["usage"] = LLMUsage.empty_usage()

            action_dict = (
                scratchpad.action.to_dict()
                if scratchpad.action
                else {"action": scratchpad.agent_response}
            )

            yield self.finish_log_message(
                log=model_log,
                data={"thought": scratchpad.thought, **action_dict},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            if not scratchpad.action:
                final_answer = scratchpad.thought
            else:
                run_agent_state = True
                # action is tool call, invoke tool
                tool_call_started_at = time.perf_counter()
                tool_name = scratchpad.action.action_name
                tool_call_log = self.create_log_message(
                    label=f"CALL {tool_name}",
                    data={},
                    metadata={
                        LogMetadata.STARTED_AT: time.perf_counter(),
                        LogMetadata.PROVIDER: tool_instances[
                            tool_name
                        ].identity.provider
                        if tool_instances.get(tool_name)
                        else "",
                    },
                    parent=round_log,
                    status=ToolInvokeMessage.LogMessage.LogStatus.START,
                )
                yield tool_call_log
                (
                    tool_invoke_response,
                    tool_invoke_parameters,
                    additional_messages,
                ) = self._handle_invoke_action(
                    action=scratchpad.action,
                    tool_instances=tool_instances,
                    message_file_ids=message_file_ids,
                )
                scratchpad.observation = tool_invoke_response
                scratchpad.agent_response = tool_invoke_response

                # TODO: convert to agent invoke message
                yield from additional_messages
                yield self.finish_log_message(
                    log=tool_call_log,
                    data={
                        "tool_name": tool_name,
                        "tool_call_args": tool_invoke_parameters,
                        "output": tool_invoke_response,
                    },
                    metadata={
                        LogMetadata.STARTED_AT: tool_call_started_at,
                        LogMetadata.PROVIDER: tool_instances[
                            tool_name
                        ].identity.provider
                        if tool_instances.get(tool_name)
                        else "",
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter()
                        - tool_call_started_at,
                    },
                )

                # update prompt tool message
                for prompt_tool in self._prompt_messages_tools:
                    self.update_prompt_message_tool(
                        tool_instances[prompt_tool.name], prompt_tool
                    )
            yield self.finish_log_message(
                log=round_log,
                data={
                    "action_name": scratchpad.action.action_name
                    if scratchpad.action
                    else "",
                    "action_input": scratchpad.action.action_input
                    if scratchpad.action
                    else "",
                    "thought": scratchpad.thought,
                    "observation": scratchpad.observation,
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            iteration_step += 1

        # yield self.create_text_message(final_answer)

        # If context is a list of dict, create retriever resource message
        if isinstance(react_params.context, list):
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
                    for ctx in react_params.context
                ],
                context="",
            )

        yield self.create_json_message(
            {
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: llm_usage["usage"].total_price
                    if llm_usage["usage"] is not None
                    else 0,
                    LogMetadata.CURRENCY: llm_usage["usage"].currency
                    if llm_usage["usage"] is not None
                    else "",
                    LogMetadata.TOTAL_TOKENS: llm_usage["usage"].total_tokens
                    if llm_usage["usage"] is not None
                    else 0,
                }
            }
        )

    def _organize_user_query(
        self, query, prompt_messages: list[PromptMessage]
    ) -> list[PromptMessage]:
        """
        Organize user query
        """
        prompt_messages.append(UserPromptMessage(content=query))

        return prompt_messages

    def _organize_prompt_messages(
        self, agent_scratchpad: list, query: str
    ) -> list[PromptMessage]:
        """
        Organize
        """
        # organize system prompt
        system_message = self._system_prompt_message

        # organize current assistant messages
        agent_scratchpad = agent_scratchpad
        if not agent_scratchpad:
            assistant_messages = []
        else:
            assistant_message = AssistantPromptMessage(content="")
            for unit in agent_scratchpad:
                if unit.is_final():
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Final Answer: {unit.agent_response}"
                else:
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Thought: {unit.thought}\n\n"
                    if unit.action_str:
                        assistant_message.content += f"Action: {unit.action_str}\n\n"
                    if unit.observation:
                        assistant_message.content += (
                            f"Observation: {unit.observation}\n\n"
                        )

            assistant_messages = [assistant_message]

        # query messages
        query_messages = self._organize_user_query(query, [])

        if assistant_messages:
            # organize historic prompt messages
            historic_messages = self.history_prompt_messages
            messages = [
                system_message,
                *historic_messages,
                *query_messages,
                *assistant_messages,
                UserPromptMessage(content="continue"),
            ]
        else:
            # organize historic prompt messages
            historic_messages = self.history_prompt_messages
            messages = [system_message, *historic_messages, *query_messages]

        # join all messages
        return messages

    def _handle_invoke_action(
        self,
        action: AgentScratchpadUnit.Action,
        tool_instances: Mapping[str, ToolEntity],
        message_file_ids: list[str],
    ) -> tuple[str, dict[str, Any] | str, list[ToolInvokeMessage]]:
        """
        handle invoke action
        :param action: action
        :param tool_instances: tool instances
        :param message_file_ids: message file ids
        :param trace_manager: trace manager
        :return: observation, meta
        """
        # action is tool call, invoke tool
        tool_call_name = action.action_name
        tool_call_args = action.action_input
        tool_instance = tool_instances.get(tool_call_name)

        if not tool_instance:
            answer = f"there is not a tool named {tool_call_name}"
            return answer, tool_call_args, []

        if isinstance(tool_call_args, str):
            try:
                tool_call_args = json.loads(tool_call_args)
            except json.JSONDecodeError as e:
                params = [
                    param.name
                    for param in tool_instance.parameters
                    if param.form == ToolParameter.ToolParameterForm.LLM
                ]
                if len(params) > 1:
                    raise ValueError("tool call args is not a valid json string") from e
                tool_call_args = {params[0]: tool_call_args} if len(params) == 1 else {}
        tool_call_args = cast(dict[str, Any], tool_call_args)
        tool_invoke_parameters = {**tool_instance.runtime_parameters, **tool_call_args}
        try:
            tool_invoke_responses = self.session.tool.invoke(
                provider_type=ToolProviderType(tool_instance.provider_type),
                provider=tool_instance.identity.provider,
                tool_name=tool_instance.identity.name,
                parameters=tool_invoke_parameters,
            )
            result = ""
            additional_messages = []  # Collect messages that need to be yielded
            for response in tool_invoke_responses:
                result += self._serialize_tool_response(response)
                if response.type in {
                    ToolInvokeMessage.MessageType.IMAGE_LINK,
                    ToolInvokeMessage.MessageType.IMAGE,
                    ToolInvokeMessage.MessageType.BLOB,
                    ToolInvokeMessage.MessageType.FILE,
                    "binary_link"
                }:
                    additional_messages.append(response)
        except Exception as e:
            result = f"tool invoke error: {e!s}"
            additional_messages = []

        return result, tool_invoke_parameters, additional_messages

    def _convert_dict_to_action(self, action: dict) -> AgentScratchpadUnit.Action:
        """
        convert dict to action
        """
        return AgentScratchpadUnit.Action(
            action_name=action["action"], action_input=action["action_input"]
        )

    def _format_assistant_message(
        self, agent_scratchpad: list[AgentScratchpadUnit]
    ) -> str:
        """
        format assistant message
        """
        message = ""
        for scratchpad in agent_scratchpad:
            if scratchpad.is_final():
                message += f"Final Answer: {scratchpad.agent_response}"
            else:
                message += f"Thought: {scratchpad.thought}\n\n"
                if scratchpad.action_str:
                    message += f"Action: {scratchpad.action_str}\n\n"
                if scratchpad.observation:
                    message += f"Observation: {scratchpad.observation}\n\n"

        return message

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
            ToolInvokeMessage.MessageType.BINARY_LINK,
        }:
            logger.info(f"Serializing IMAGE/LINK/BINARY response: {response.type}")
            if hasattr(response.message, "text"):
                file_info = cast(
                    ToolInvokeMessage.TextMessage, response.message
                ).text
                try:
                    tool_file_id = file_info.split("/")[-1].split(".")[0]
                    serialized = json.dumps({
                        "related_id": tool_file_id,
                        "transfer_method": "tool_file",
                    }, ensure_ascii=False)
                    logger.info(f"Serialized file link: {serialized}")
                    result += serialized
                except Exception as e:
                    logger.info(f"Failed to parse tool_file_id from {file_info}: {e}")
                    result += file_info
        elif response.type == ToolInvokeMessage.MessageType.JSON:
            text = json.dumps(
                cast(
                    ToolInvokeMessage.JsonMessage, response.message
                ).json_object,
                ensure_ascii=False,
            )
            result += f"tool response: {text}."
        elif response.type == ToolInvokeMessage.MessageType.BLOB:
            logger.info("Serializing BLOB response")
            if hasattr(response.message, "text"):
                file_info = cast(
                    ToolInvokeMessage.TextMessage, response.message
                ).text
                try:
                    tool_file_id = file_info.split("/")[-1].split(".")[0]
                    serialized = json.dumps({
                        "related_id": tool_file_id,
                        "transfer_method": "tool_file",
                    }, ensure_ascii=False)
                    logger.info(f"Serialized BLOB: {serialized}")
                    result += serialized
                except Exception as e:
                    logger.info(f"Failed to parse tool_file_id from BLOB info {file_info}: {e}")
                    result += "Generated file ... "
            else:
                result += "Generated file ... "
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

