import shutil
import json
import os
import sys
import os.path as osp
from datetime import datetime

from aider.models import Model
from aider.io import InputOutput
from aider.coders import Coder

from perform_experiments import perform_experiments

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        log_file=False,
    ):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f'Folder {folder_name} already exists.'

    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)

    with open(osp.join(base_dir, 'run_0', 'final_info.json'), 'r') as f:
        baseline_results = json.load(f)

    baseline_results = {k: v['means'] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, 'experiment.py')
    vis_file = osp.join(folder_name, 'plot.py')
    notes = osp.join(folder_name, 'notes.txt')

    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")

    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log

    print_time()
    print(f"*Starting idea: {idea_name}*")
    ## PERFORM EXPERIMENTS
    fnames = [exp_file, vis_file, notes]
    io = InputOutput(
        yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
    )

    main_model = Model(model)

    coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
    
    print_time()
    print(f"*Starting Experiments*")
    try:
        success = perform_experiments(idea, folder_name, coder, baseline_results)
    except Exception as e:
        print(f"Error during experiments: {e}")
        print(f"Experiments failed for idea {idea_name}")
        return False
    
    if not success:
        print(f"Experiments failed for idea {idea_name}")
        return False
    else:
        print(f"Experiments completed for idea {idea_name}")
    return True
