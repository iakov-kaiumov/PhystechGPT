"""Bot configuration parameters."""

import yaml
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / 'config.yml', "r") as f:
    config = yaml.safe_load(f)

# Bot version.
version = 61

# Telegram Bot API token.
telegram_token = config["telegram_token"]

# OpenAI API key.
openai_api_key = config["openai_api_key"]

# The list of Telegram usernames allowed to chat with the bot.
# If empty, the bot will be available to anyone.
telegram_usernames = set(config["telegram_usernames"])

# The list of Telegram group ids, whose members are allowed to chat with the bot.
# If empty, the bot will only be available to `telegram_usernames`.
telegram_chat_ids = config.get("telegram_chat_ids", [])

# The maximum number of previous messages
# the bot will remember when talking to a user.
max_history_depth = config.get("max_history_depth", 3)

# Where to store the chat context file.
persistence_path = config["persistence_path"]
