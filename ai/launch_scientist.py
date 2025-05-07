import json
import openai
import os
import os.path as osp
from datetime import datetime
from generate_ideas import generate_ideas
from do_ideas import do_idea


NUM_REFLECTIONS = 3

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError('Missing OpenAI API key')
client_model = 'gpt-4o-mini'
client = openai.OpenAI(api_key=OPENAI_API_KEY)



base_dir = osp.abspath('./templates')
results_dir = osp.abspath('./results')

skip_generation = True

ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=skip_generation,
        max_num_generations=2,
        num_reflections=NUM_REFLECTIONS,
    )

print('Done generating ideas')

with open(osp.join(base_dir, 'ideas.json'), 'w') as f:
    json.dump(ideas, f, indent=4)

novel_ideas = [idea for idea in ideas]

for idea in novel_ideas:
    print(f"Processing idea: {idea['Name']}")
    try:
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            client_model,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    except Exception as e:
        print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

print("All ideas evaluated.")
