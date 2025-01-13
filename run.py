import logging
import os

from agent.agent import AndroidAgent

GOAL = """
1. Open the Chrome app
2. Enter airbnb.com into the search bar
3. Tap enter button
4. Tap the first listing you see

Your task is done when the screen contains information about the listing.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(name)s] (%(levelname)s) %(message)s")

if __name__ == "__main__":
    device_id = None  # or "emulator-5554", etc.

    agent = AndroidAgent(device_id=device_id,
                         model_backend="openai",
                         model="gpt-4o-mini",
                         max_loops=30,
                         screen_wait_seconds=4,
                         temperature=0.2,)

    # Close all apps
    agent.ui_manager.reset()
    # Execute the agent
    result = agent.run(GOAL)
    print("Result:", result)