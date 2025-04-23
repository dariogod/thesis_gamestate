import os

base_path = 'outputs/sn-gamestate/2025-04-23/13-16-56/eval/pred/SoccerNetGS-challenge/tracklab'
data_path = 'data/SoccerNetGS/gamestate-2024/challenge'

for file in os.listdir(base_path):
    if file.startswith('action_') and file.endswith('.json'):
        folder_to_remove = file.split('.json')[0]
        import shutil
        shutil.rmtree(f"{data_path}/{folder_to_remove}")

