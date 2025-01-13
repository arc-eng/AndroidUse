# Android LLM Automation Agent

This project demonstrates how to create an LLM-powered Android automation agent using [uiautomator2](https://github.com/openatx/uiautomator2) and [LangChain](https://github.com/hwchase17/langchain).

### Files

- **`models.py`**  
  Contains Pydantic models describing the possible UI actions (tap, input_text, press_key) and the JSON schema for the LLM’s output.

- **`manager.py`**  
  Provides `UIAutomatorManager`, which encapsulates device-specific operations like dumping the UI hierarchy, tapping, inputting text, and pressing keys.

- **`runnables.py`**  
  Defines LangChain `Runnable` classes that orchestrate actions:
    1. **HierarchyRunnable** to dump and parse the UI hierarchy.
    2. **DecideActionRunnable** to query an LLM for the next action.
    3. **PerformActionRunnable** to execute that action on the device.
    4. **LoopRunnable** to repeat the cycle until completion.

- **`agent.py`**  
  Defines an `AndroidAgent` class which ties everything together. It instantiates the runnables and provides a single `run(user_goal)` method that loops until the LLM decides it’s done.

## Installation & Setup

1. **Clone or copy** this repository.
2. **Install Dependencies**:
    ```bash
    poetry install
    ```

3. **Connect an Android device**:
    - Enable USB debugging on your Android phone or use an emulator.
    - Ensure you have `adb` installed and your device is recognized via `adb devices`.

4. **Set your OpenAI API key**:
    - Either store it in an environment variable:
      ```bash
      export OPENAI_API_KEY="your-secret-key"
      ```
    - Or provide it directly when creating the agent.

## Usage

In Python:

```python
import logging
import os

from agent.agent import AndroidAgent

GOAL = """
Open YouTube, search for podcast, and play the first video. 
Find the name of the video creator, then Google it (use Android's native search, not Chrome).
Respond with a list of links from the search results.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(name)s] (%(levelname)s) %(message)s")

if __name__ == "__main__":
    # Replace with your actual OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    device_id = None  # or "emulator-5554", etc.

    agent = AndroidAgent(openai_api_key, device_id=device_id, max_loops=30)

    # Close all apps
    agent.ui_manager.reset()
    # Execute the agent
    result = agent.run(GOAL)
    print("Result:", result)

```