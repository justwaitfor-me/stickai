from consolemenu import *
from consolemenu.items import *

from utils import get_config, project

from player import llm_player, dqn_player, human_player, random_player

# Load configuration settings from the config
cp = get_config()["project"]

project(cp)

# Create the menu
menu = ConsoleMenu(formatter=MenuFormatBuilder(), clear_screen=False, prologue_text="Select a player agent to start the game!")

# Create items
llm_player_ = FunctionItem("Large Language Modal", llm_player)

dqn_player_ = FunctionItem("Deep Q-Network", dqn_player)

human_player_ = FunctionItem("Human Player", human_player)

random_player_ = FunctionItem("Random Moves", random_player)


# Add the items to the menu
menu.append_item(llm_player_)
menu.append_item(dqn_player_)
menu.append_item(human_player_)
menu.append_item(random_player_)

# Show the menu
menu.show()

