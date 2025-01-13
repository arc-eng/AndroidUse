import os
import logging

from langchain_openai import ChatOpenAI

# If you're using local Hugging Face:
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

from .manager import UIAutomatorManager
from .runnables import (
    HierarchyRunnable,
    DecideActionRunnable,
    PerformActionRunnable,
    LoopRunnable,
)

logger = logging.getLogger(__name__)

class AndroidAgent:
    """
    An Android automation agent that uses an LLM to interpret a user's goal
    and interact with an Android device via UIAutomator2.
    """

    def __init__(
            self,
            model: str = "gpt-4o",
            device_id: str = None,
            screen_wait_seconds: int = 4,
            max_loops: int = 8,
            openai_api_key: str = None,
            model_backend: str = "openai",  # "openai" or "local"
            local_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.3",
            temperature: float = 0.2
    ):
        """
        :param model: For OpenAI backend, this is the OpenAI model (e.g. "gpt-4o").
                      For local, this might not be used, or it can be a reference name.
        :param device_id: The Android device/emulator ID (e.g., "emulator-5554").
        :param screen_wait_seconds: How long to wait (in seconds) after a UI action
                                    before dumping the hierarchy again.
        :param max_loops: Maximum number of action loops before stopping.
        :param openai_api_key: If using OpenAI, your API key. Otherwise, it can be None.
        :param model_backend: "openai" or "local". Determines which LLM to load.
        :param local_model_name_or_path: If using local, a HuggingFace model name
                                         (e.g. "tiiuae/falcon-7b-instruct")
                                         or a path to a local checkpoint.
        :param temperature: The temperature for generation (both local & OpenAI).
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_backend = model_backend
        self.local_model_name_or_path = local_model_name_or_path
        self.model = model
        self.temperature = temperature

        self.device_id = device_id
        self.max_loops = max_loops
        self.screen_wait_seconds = screen_wait_seconds

        # Set up the UI manager
        self.ui_manager = UIAutomatorManager(device_id=self.device_id)

        # Create the runners
        self.hierarchy_runner = HierarchyRunnable(self.ui_manager)
        self.llm = self._initialize_llm()
        self.decide_runner = DecideActionRunnable(self.llm)
        self.perform_runner = PerformActionRunnable(self.ui_manager)

        # Build the "flow" runner
        self.loop_runner = LoopRunnable(
            hierarchy_runner=self.hierarchy_runner,
            decide_runner=self.decide_runner,
            perform_runner=self.perform_runner,
            max_loops=self.max_loops,
            screen_wait_seconds=self.screen_wait_seconds,
        )

    def _initialize_llm(self):
        """
        Depending on model_backend, return the correct LLM instance (OpenAI or Local).
        """
        if self.model_backend == "openai":
            logger.info(f"Initializing OpenAI LLM: {self.model}")
            return ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model_name=self.model,
                temperature=self.temperature
            )

        elif self.model_backend == "local":
            logger.info("Initializing local Hugging Face model...")

            if not self.local_model_name_or_path:
                raise ValueError("You must provide local_model_name_or_path when model_backend='local'.")

            # Load Mistral tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.local_model_name_or_path, use_auth_token=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name_or_path,
                device_map="auto",  # Automatically place on GPU/MPS if available
                torch_dtype="auto",  # Use half-precision (float16) if supported
                use_auth_token=True
            )

            # Create the Hugging Face pipeline
            hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=50,  # Limit number of generated tokens
                temperature=self.temperature,
                do_sample=True
            )

            # Wrap the pipeline with LangChain's HuggingFacePipeline
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            return llm

        else:
            raise ValueError(f"Unknown model_backend: {self.model_backend}")


    def run(self, user_goal: str) -> str:
        """
        Executes the user_goal against the Android device, returning
        the final output of the LLM or the reason the process stopped.
        """
        logger.info(f"Starting AndroidAgent with backend '{self.model_backend}' and goal: {user_goal}")
        result = self.loop_runner.invoke({"user_goal": user_goal})
        return result
