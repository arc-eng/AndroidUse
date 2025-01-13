# my_agent/manager.py
import logging
import uiautomator2 as u2
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class UIAutomatorManager:
    def __init__(self, device_id=None):
        if device_id:
            self.device = u2.connect(device_id)
        else:
            self.device = u2.connect()

    def dump_ui_hierarchy_xml(self) -> str:
        return self.device.dump_hierarchy()

    def parse_ui_hierarchy(self, xml_content: str) -> list:
        root = ET.fromstring(xml_content)
        elements = []
        for node in root.iter('node'):
            elements.append({
                "resource_id": node.attrib.get("resource-id"),
                "text": node.attrib.get("text"),
                "content_desc": node.attrib.get("content-desc"),
                "class": node.attrib.get("class"),
                "bounds": node.attrib.get("bounds"),
                "clickable": node.attrib.get("clickable"),
            })
        return elements

    def tap(self, resource_id=None, text=None, content_desc=None):
        if resource_id:
            logger.debug(f"Tapping resource_id={resource_id}")
            self.device(resourceId=resource_id).click_exists(timeout=3)
        elif text:
            logger.debug(f"Tapping text={text}")
            self.device(text=text).click_exists(timeout=3)
        elif content_desc:
            logger.debug(f"Tapping content_desc={content_desc}")
            self.device(description=content_desc).click_exists(timeout=3)
        else:
            logger.warning("No valid identifier to tap.")

    def input_text(self, text):
        logger.debug(f"Inputting text: {text}")
        self.device.send_keys(text, clear=True)

    def press_key(self, key):
        self.device.press(key)

    def reset(self):
        """
        Attempt to close all running apps.
        """
        logger.info("Resetting device by closing all apps and going to home screen")

        self.device.app_stop_all()
        self.device.press("home")
