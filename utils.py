from typing import Dict, List
from dotenv import load_dotenv
from art import tprint
import datetime
import logging
import json
import yaml
import re
import os

def get_config() -> Dict:
    """Loads the configuration file."""
    with open('conf.yml', 'r') as file:
        return yaml.safe_load(file)

def debug():
    if get_config()["settings"]["debug"]:
        logging.basicConfig(level=logging.DEBUG)
        return True
    else:
        logging.basicConfig(level=logging.INFO)
        return False

def clear_screen() -> None:
    """Clear Terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def project(project: dict) -> None:
    """Prints start screen and ascii art"""
    clear_screen()
    tprint(project["name"])
    print(project["version"])
    print(f"Author: {project["author"]}")
    print("#"*int(os.get_terminal_size().columns))
    print("\n")

def get_key() -> str:
    """Loads the API key from the .env file."""
    load_dotenv()
    try:
        return os.getenv("API_KEY", "")
    except KeyError:
        return input("API Key not found. Please enter your API key: ")
    except Exception as e:
        print(f"An error occurred while retrieving the API key: {e}")
        return ""

def build_prompt(game: str, round: int, max_sticks: int) -> str:
    """Creates a JSON-formatted string as input for an AI."""
    return json.dumps({
        "current_sticks": game,
        "current_round": round,
        "max_sticks": max_sticks,
    })
    
def read_file(path: str) -> str:
    """Reads a file and returns its content."""
    with open(path, "r") as file:
        return file.read()
    
def save_text(file_path: str, text: str, mode: str = "w") -> None:
    """Saves text to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as file:
        file.write(text)
    
def parse_ai_response(response_str: str) -> Dict:
    """Parses the AI response and returns a JSON object."""
    try:
        response_str = re.sub(r"```(?:json)?\s*|```", "", response_str).strip()
        response_json = json.loads(response_str)
        if "explanation" in response_json:
            response_json["explanation"] = response_json["explanation"].replace("\n", " ")
        return response_json
    except json.JSONDecodeError:
        return {}
    
def log_llm_explanations(explanations: List[Dict[str, str]], uid: str, round: int) -> None:
    """Saves AI explanations."""
    current_time = datetime.now().strftime('%H:%M:%S')
    text = f"(Player {round}) {current_time}\n"
    text += "\n".join(f"({exp[0]}) {exp[1]['move']}: {exp[1]['explanation']}" for exp in explanations) + "\n\n"
    
    save_text(f"logs/{uid}/llm/explanations.log", text, mode="a")
    
def calculate_intervals(episodes):
    max_points = 1000  # Maximal so viele Punkte für Grafik und Speicherung
    update_every = max(1, episodes // max_points)
    save_every = max(100, episodes // max_points)
    return update_every, save_every
