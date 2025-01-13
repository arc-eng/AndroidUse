# my_agent/models.py
import json
from typing import Optional, Union, Literal

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


# ──────────────────────────────────────────────────────────────────────────────
# 1. Define Pydantic Models for Our Actions
# ──────────────────────────────────────────────────────────────────────────────

class TapAction(BaseModel):
    action: Literal["tap"] = "tap"
    resource_id: Optional[str] = None
    text: Optional[str] = None
    content_desc: Optional[str] = None

class InputTextAction(BaseModel):
    action: Literal["input_text"] = "input_text"
    resource_id: str = Field(..., description="The resource ID of the input field")
    text: str = Field(..., description="The text to input")

class PressKeyAction(BaseModel):
    action: Literal["press_key"] = "press_key"
    key: str

UIActionUnion = Union[TapAction, InputTextAction, PressKeyAction]


class UIActionOutput(BaseModel):
    """
    - action_type: either 'ui_action' or 'final'
    - action_content:
      * If 'ui_action', one of TapAction, InputTextAction, PressKeyAction
      * If 'final', a plain string
    - explanation: optional human-readable text
    """
    action_type: Literal["ui_action", "final"]
    action_content: Optional[Union[UIActionUnion, str]] = None
    explanation: Optional[str] = None


# Create a parser that can parse JSON output into our UIActionOutput model
pydantic_parser = PydanticOutputParser(pydantic_object=UIActionOutput)
