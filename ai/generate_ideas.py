import json
import os
import os.path as osp
from prompts.prompts import idea_first_prompt, idea_reflection_prompt
from llm import extract_json_between_markers, get_response_from_llm


def generate_ideas(base_dir,
                   client,
                   model,
                   skip_generation=False,
                   max_num_generations=10,
                   num_reflections=5,):
    
    if skip_generation:
        print('Skipping idea generation')
        with open(osp.join(base_dir, 'ideas.json'), 'r') as f:
            ideas = json.load(f)
        return ideas

    with open(osp.join(base_dir, 'prompt.json'), 'r') as f:
        prompt = json.load(f)

    idea_system_prompt = prompt['system']

    idea_str_archive = []
    if osp.exists(osp.join(base_dir, 'seed_ideas.json')):
        with open(osp.join(base_dir, 'seed_ideas.json'), 'r') as f:
            seed_ideas = json.load(f)
    else:
        seed_ideas = {}

    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))
    
    with open(osp.join(base_dir, 'experiment.py'), 'r') as f:
        code = f.read()

    for _ in range(max_num_generations):
        print()
        print(f'Generating idea {_ + 1}/{max_num_generations}')
        prev_ideas_string = '\n\n'.join(idea_str_archive)

        msg_history = []
        print(f'Iteration 1/{num_reflections}')

        text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
        json_output = extract_json_between_markers(text)
        assert json_output is not None, 'Failed to extract JSON from LLM output'
        print(json_output)

        for j in range(num_reflections - 1):
            print(f'Iteration {j + 2}/{num_reflections}')
            text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, 'Failed to extract JSON from LLM output'
            print(json_output)
            
            if 'I am done' in text:
                print(f'Idea generation converged after {j + 2} iterations.')
                break

        idea_str_archive.append(json.dumps(json_output))

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    return ideas


