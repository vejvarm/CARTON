import json
from pathlib import Path
from tqdm import tqdm


def _load_data(read_folder: Path) -> dict[str: dict]:
    csqa_files = read_folder.glob("**/*.json")

    csqa_data = {}
    total_conv = 0
    for path in csqa_files:
        folder = path.parent.name
        file = path.name
        if folder not in csqa_data:
            csqa_data[folder] = {}

        with open(path, 'r', encoding='utf8') as json_file:
            csqa_data[folder][file] = json.load(json_file)
        total_conv += 1
    print(f'Done, {len(csqa_data)} folders loaded!')
    return csqa_data


def clean_up_qa_entries(read_folder: Path, write_folder: Path):
    csqa_data = _load_data(read_folder)

    for folder in tqdm(csqa_data.keys()):
        csqa_folder = csqa_data[folder]
        for file in csqa_folder.keys():
            conversation = csqa_folder[file]

            for turn in conversation:
                turn["utterance"] = turn["utterance"].replace(" ? ", "? ").strip()

            # create path
            folder_path = write_folder.joinpath(folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            # write conversation
            with folder_path.joinpath(file).open('w', encoding='utf8') as json_file:
                json.dump(conversation, json_file, ensure_ascii=False, indent=4)


def annotate_position(read_folder: Path, write_folder: Path):
    csqa_data = _load_data(read_folder)

    # NOTE: conversation == one QA_*.json file
    for folder in tqdm(csqa_data.keys()):
        csqa_folder = csqa_data[folder]
        for file in csqa_folder.keys():
            # get conversation
            conversation = csqa_folder[file]

            for i, turn in enumerate(conversation):
                if i % 2 == 0:
                    turn["turn_position"] = i//2

            # create path
            folder_path = write_folder.joinpath(folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            # write conversation
            with folder_path.joinpath(file).open('w', encoding='utf8') as json_file:
                json.dump(conversation, json_file, ensure_ascii=False, indent=4)


def extract_simple(read_folder: Path, write_folder: Path):
    csqa_data = _load_data(read_folder)

    for folder in tqdm(csqa_data.keys()):
        csqa_folder = csqa_data[folder]
        for file in csqa_folder.keys():
            # get conversation
            conversation = csqa_folder[file]

            conversation_simple = []
            for i in range(0, len(conversation), 2):
                q_type = conversation[i]["question-type"]
                q_utterance = conversation[i]["utterance"]
                if "Simple Question (Direct)" in q_type:
                    conversation_simple.extend(conversation[i:i + 2])
                # if "Simple Question (Coreferenced)" in q_type and q_utterance.startswith(("Yes", "No, ")):
                #     continue  # NOTE: IGNORE "Yes"/"No, I meant" types of questions
                # if "Simple Question" in q_type:
                #     conversation_simple.extend(conversation[i:i + 2])

            # create path
            folder_path = write_folder.joinpath(folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            # write conversation
            with folder_path.joinpath(file).open('w', encoding='utf8') as json_file:
                json.dump(conversation_simple, json_file, ensure_ascii=False, indent=4)