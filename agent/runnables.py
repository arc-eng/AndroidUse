# my_agent/runnables.py
import logging
import time

from langchain.schema.runnable import Runnable, RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .manager import UIAutomatorManager
from .models import pydantic_parser, UIActionOutput

logger = logging.getLogger(__name__)


class HierarchyRunnable(Runnable):
    def __init__(self, ui_manager: UIAutomatorManager):
        self.ui_manager = ui_manager

    def invoke(self, input_data, config: RunnableConfig = None) -> dict:
        logger.debug("Dumping UI hierarchy...")
        xml_content = self.ui_manager.dump_ui_hierarchy_xml()
        elements = self.ui_manager.parse_ui_hierarchy(xml_content)
        return {**input_data, "ui_elements": xml_content}


class DecideActionRunnable(Runnable):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def invoke(self, input_data: dict, config: RunnableConfig = None) -> dict:
        user_goal = input_data.get("user_goal", "No goal provided.")
        ui_elements = input_data.get("ui_elements", [])

        # Incorporate history
        history_list = input_data.get("history", [])
        history_text = "\n".join(history_list)

        ui_text = "\n".join([str(el) for el in ui_elements])

        format_instructions = pydantic_parser.get_format_instructions()

        system_prompt = f"""
You are an Android UI automation agent.

# Instructions
You perform tasks for the user by interacting with the UI.
You can interact with the UI in three ways:
1) Tap an element.
2) Input text.
3) Press a key.


# User Request
{user_goal}

# Output Format
You must output valid JSON following this schema:

{format_instructions}

Examples:
1) UI action (tap):
{{
  "action_type": "ui_action",
  "action_content": {{
    "action": "tap",
    "text": "Gmail"
  }},
  "explanation": "Tapping Gmail app to open it"
}}

2) UI action (input_text):
{{
  "action_type": "ui_action",
  "action_content": {{
    "action": "input_text",
    "text": "Hello world",
    "resource_id": "com.example:id/text_input_search"
  }},
  "explanation": "Typing 'Hello world', because it's a friendly thing to do"
}}

3) UI action (press_key):
{{
  "action_type": "ui_action",
  "action_content": {{
    "action": "press_key",
    "key": "back"
  }},
  "explanation": "Pressing back key, because this screen is not needed"
}}

4) Final:
{{
  "action_type": "final",
  "action_content": "<Your response to the user's request>"
  "explanation": "Finished <task> in <n> steps."
}}

DO NOT add any extra keys beyond the schema.
Only respond with JSON. If you don't follow the schema, the system will stop.

"""

        human_prompt = f"""
# History of your previous actions as part of this task (if any)
{history_text}
# The current UI hierarchy of the screen:

```xml
{ui_text}
```

# Your Task

Verify if what you see in the UI is an expected result of the last step
and decide what to do next.
Only respond with JSON. If you don't follow the schema, the system will stop.

"""


        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = self.llm.invoke(messages)
        # If response is a string, it's from a local model
        if isinstance(response, str):
            raw_content = response.split("Assistant:")[1].strip("\"").strip(",").strip()
        else:
            raw_content = response.content

        # Parse using the PydanticOutputParser
        try:
            parsed_output: UIActionOutput = pydantic_parser.parse(raw_content)

            # Convert to a dict for PerformActionRunnable
            if parsed_output.action_type == "ui_action":
                # action_content is a union model or possibly a string
                if isinstance(parsed_output.action_content, str):
                    # If the LLM messed up and gave a string
                    action_dict = {}
                else:
                    action_dict = parsed_output.action_content.dict()

                return {
                    "action_type": parsed_output.action_type,
                    "action_content": action_dict,
                    "explanation": parsed_output.explanation or ""
                }
            else:
                # final
                return {
                    "action_type": "final",
                    "action_content": parsed_output.action_content,
                    "explanation": parsed_output.explanation or ""
                }

        except Exception as e:
            logger.warning(f"Failed to parse LLM output: {e}")
            return {
                "action_type": "final",
                "action_content": f"Invalid JSON: {raw_content}",
                "explanation": "Parsing error"
            }


class PerformActionRunnable(Runnable):
    def __init__(self, ui_manager: UIAutomatorManager):
        self.ui_manager = ui_manager

    def invoke(self, input_data: dict, config: RunnableConfig = None) -> dict:
        action_type = input_data.get("action_type")
        explanation = input_data.get("explanation", "")
        action_content = input_data.get("action_content", {})

        if action_type == "final":
            logger.info("LLM indicated final action. Stopping workflow.")
            return {"done": True, "message": action_content or explanation}

        if action_type != "ui_action":
            msg = f"No recognized action_type: {action_type}, stopping."
            logger.warning(msg)
            return {"done": True, "message": msg}


        action = action_content.get("action")

        logger.debug(f"Performing UI action: {action_content}")
        logger.info(explanation)

        if action == "tap":
            rid = action_content.get("resource_id")
            txt = action_content.get("text")
            desc = action_content.get("content_desc")
            self.ui_manager.tap(rid, txt, desc)

        elif action == "input_text":
            txt = action_content.get("text")
            resource_id = action_content.get("resource_id")
            if resource_id:
                self.ui_manager.tap(resource_id=resource_id)
            time.sleep(2)
            self.ui_manager.input_text(txt)

        elif action == "press_key":
            key = action_content.get("key")
            self.ui_manager.press_key(key)

        else:
            msg = f"Unrecognized UI action: {action}"
            logger.warning(msg)
            return {"done": False, "message": msg}
        return {"done": False, "message": explanation or "Action performed."}


class LoopRunnable(Runnable):
    def __init__(
            self,
            hierarchy_runner: HierarchyRunnable,
            decide_runner: DecideActionRunnable,
            perform_runner: PerformActionRunnable,
            max_loops: int = 5,
            screen_wait_seconds: int = 4
    ):
        self.hierarchy_runner = hierarchy_runner
        self.decide_runner = decide_runner
        self.perform_runner = perform_runner
        self.max_loops = max_loops
        self.screen_wait_seconds = screen_wait_seconds

    def invoke(self, input_data, config: RunnableConfig = None):
        # Keep a short memory of the entire conversation
        state = dict(input_data)
        state.setdefault("history", [])

        for i in range(self.max_loops):
            # 1. Dump UI
            state = self.hierarchy_runner.invoke(state, config)

            # 2. Decide next action
            decision = self.decide_runner.invoke(state, config)
            explanation = decision.get("explanation", "")

            # 3. Perform action
            result = self.perform_runner.invoke(decision, config)

            # 4. Log this step to history
            step_summary = f"Step {i+1}: {explanation}"
            state["history"].append(step_summary)

            if result.get("done"):
                return result.get("message", "Final result: Task ended.")
            else:
                logger.debug(f"Waiting for {self.screen_wait_seconds} seconds...")
                time.sleep(self.screen_wait_seconds)

        return "Max loops reached without final answer. Stopping."
