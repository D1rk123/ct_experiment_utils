import os
import re
import shutil
import sys
import datetime
from pathlib import Path

def make_new_experiment_folder(save_folder, name="", scripts_src_path=None):
    save_folder = Path(save_folder).expanduser().resolve()
    base_name = f"{datetime.datetime.now():%Y-%m-%d}_{name}_"
    existing_folders = save_folder.glob(f"{base_name}*")
    existing_numbers = [re.search(base_name+r'\d+$', str(f)) for f in existing_folders]
    highest_existing = max([int(n.group()[len(base_name):]) for n in existing_numbers if n is not None] + [0])
    
    # In a PyTorch Lightning multi GPU setup, only create a folder once
    # Use Local Rank to check if you're the main instance running of the script
    if(os.environ.get("LOCAL_RANK", 0) == 0):
        folder_path = save_folder / Path(base_name+str(highest_existing+1))
        folder_path.mkdir(exist_ok=False, parents=True)
        add_scripts_folder(folder_path, scripts_src_path)
        add_console_argv(folder_path)
    else:
        folder_path = save_folder / Path(base_name+str(highest_existing))
    return(folder_path)
    
def add_scripts_folder(dst_path, src_path=None):
    if src_path is None:
        src_path = Path(__file__).parent.parent
    else:
        src_path = Path(src_path).expanduser().resolve()
        
    script_folder = Path(dst_path).expanduser().resolve() / "scripts"
    script_folder.mkdir(exist_ok=False, parents=False)
    script_files = src_path.glob("**/*.py")
    for script_file in script_files:
        new_path = script_folder / script_file.relative_to(src_path)
        new_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(script_file, new_path)
    
def add_console_argv(folder_path):  
    with open(folder_path / "console_argv.txt", "w") as outfile:
        outfile.write(" ".join(sys.argv))
        
def print_and_log(log_file, message):
    print(message)
    log_file.write(str(message)+"\n")
