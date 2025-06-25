from consolemenu import SelectionMenu
import random
import time
import uuid

from utils import clear_screen as clear
from utils import get_config, debug

from dqn_agent import NimGame
from dqn_agent import load_nim_agent, evaluate_nim_agent, train_nim_agent, save_nim_agent
from llm_agent import get_llm_move

def initialize_game():
    global sticks, counter, stick, uid
    config = get_config()
    sticks = config["game"]["max_sticks"]
    counter = 0
    stick = config["game"]["stick"]
    
    uid = uuid.uuid4()
    
    clear()
    
    if debug():
        print("[DEBUG] Game initialized with", sticks, "sticks.")
        print("[DEBUG] UUID for this game:", uid)
        
def end_game(won:bool, counter:int, uid:str, sticks:int = 0, msg:str = None):
    clear()
    
    print(f"[INFO] Game ended after {counter} rounds.")
    if debug():
        print("[DEBUG] Sticks remaining:", sticks)
        print("[DEBUG] Game won:", won)
        print(f"[INFO] Game UUID: {uid}")
        print()
        
    if msg == None:
        if won:
            print(f"You won with {sticks} sticks remaining!")
            
        else:
            print("You lost! Better luck next time.")
            
    else:
        print(f"{msg}")
    
    if debug():
        print("[DEBUG] Game teardown complete.")

def llm_player():
    global sticks, counter, stick, uid
    initialize_game()
    
    max_sticks = sticks
    player = None
    last_explaination = None
    taken_sticks = ""
    
    while True:
        clear()
        counter += 1
        
        if player is not None:
            taken_sticks = f" [-{player}]"
        
        if last_explaination is not None:
            print(f"[INFO] AI Explanation: {last_explaination}\n")
            last_explaination = None

        print(f"Round {counter}")
        print(f"{stick*sticks} {taken_sticks}")

        try:
            if counter % 2:
                player = int(input(f"You: "))
                last_explaination = None
            else:
                ai = get_llm_move(sticks, counter, max_sticks)
                try:
                    player = ai.get("sticks_to_take", 0)
                    last_explaination = ai.get("explanation", "")
                except KeyError:
                    print("[ERROR] Invalid AI response format. Please try again.")
                    counter -= 1
                    continue
                
            player_name = "You" if counter % 2 else "AI"
            
            if player < 0 or player > 3:
                counter -= 1
                continue
                
        except:
            counter -= 1
            
        else:
            sticks -= player
            
        if sticks <= 0:
            end_game(True, counter, uid, msg=f"{player_name} lost in round {counter}!")
            break
        
            

def dqn_player():
    global sticks, counter, stick, uid
    initialize_game()
    
    while True:
        dqn_player_menu = SelectionMenu(["Train", "Play", "Evaluate"], show_exit_option=True, prologue_text="Select an option for DQN Player:", clear_screen=False)
        dqn_player_menu.show()
        selected_option = dqn_player_menu.selected_option
        
        clear()
        
        if selected_option == 0:
            # Train the agent
            try:
                episodes = int(input("Enter number of episodes to train the agent (default 2000000): "))
                if episodes <= 0:
                    episodes = 2000000
            except ValueError:
                episodes = 2000000
                
            trained_agent, win_rates = train_nim_agent(episodes=episodes, initial_sticks=sticks)
            
            # Save the agent
            save_nim_agent(trained_agent, "nim_agent.json")
            
            print("waiting for agent to finish training...")
            time.sleep(2)  # Simulate waiting time for training to complete
            clear()
            
            print(f"[INFO] Agent trained for {episodes} episodes.")
            
            
        elif selected_option == 1:
            try:
                # Load the agent
                trained_agent = load_nim_agent("nim_agent.json")
                
                game = NimGame(sticks)
                state = game.reset()
                
                while not game.game_over:
                    counter += 1
                    # Player's turn
                    try:
                        print(f"{stick*game.current_state()} [-{ai_move}]")
                    except NameError:
                        print(f"{stick*game.current_state()}")
                        
                    while True:
                        try:
                            player_move = int(input("You: "))
                            if player_move in game.get_valid_actions():
                                break
                            else:
                                print("[ERROR] Invalid move. Please choose between 1 and 3 sticks.")
                        except ValueError:
                            print("[ERROR] Invalid input. Please enter a number.")
                    
                    state, reward, done = game.step(player_move)
                    if done:
                        end_game(False, counter, uid, sticks=game.current_state())
                        break
                    
                    # AI's turn
                    valid_actions = game.get_valid_actions()
                    ai_move = trained_agent.get_action(state, valid_actions, training=False)
                    state, reward, done = game.step(ai_move)
                    if debug():
                        print(f"[DEBUG] AI took {ai_move} sticks.")
                    if done:
                        end_game(True, counter, uid, sticks=game.current_state())
                        break
                    
            except FileNotFoundError:
                print("[ERROR] Agent file not found. Please train an agent first.")
                
        elif selected_option == 2:
            try:
                # Load the agent
                trained_agent = load_nim_agent("nim_agent.json")
                # Evaluate the agent
                clear()
                print("\n[INFO] Evaluation Results:")
                evaluate_nim_agent(trained_agent)
            except FileNotFoundError:
                print("[ERROR] Agent file not found. Please train an agent first.")
                
        elif selected_option == 3:
            if debug():
                print("[DEBUG] Exiting DQN Player.")
            break
                
        else:
            clear()
            print("[ERROR] Invalid option selected for DQN Player.")
        

def human_player():
    global sticks, counter, stick, uid
    initialize_game()
    
    while True:
        clear()
        counter += 1

        print(f"Round {counter}")
        print(stick*sticks)

        try:
            if counter % 2:
                player_name = "You"
            else:
                player_name = "Player 2"
                
            player = int(input(f"{player_name}: "))
                
            if player < 0 or player > 3:
                counter -= 1
                continue
                
        except:
            counter -= 1
            
        else:
            sticks -= player
            
        if sticks <= 0:
            end_game(True, counter, uid, msg=f"{player_name} lost in round {counter}!")
            break
        
            

def random_player():
    global sticks, counter, stick, uid
    initialize_game()
    player = None
    
    while True:
        clear()
        counter += 1

        print(f"Round {counter}")
        print(stick*sticks, end="")
        
        if player is not None:
            print(f" [-{player}]")
        else:
            print()

        try:
            if counter % 2:
                player_name = "You"
                player = int(input(f"{player_name}: "))
                    
                if player < 0 or player > 3:
                    counter -= 1
                    continue
            else:
                player_name = "Randomizer"
                player = random.randint(1, 3)
                
        except:
            counter -= 1
            
        else:
            sticks -= player
            
        if sticks <= 0:
            end_game(True, counter, uid, msg=f"{player_name} lost in round {counter}!")
            break
