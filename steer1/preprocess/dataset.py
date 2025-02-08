import csv

def prepare_ours_jailbreak_dataset(
    path: str = "dataset/test.json",
    chat_template: str = None,
):
    # Pre-tokenize and create DataLoader
    with open(path, "r") as f:
        test_pairs = json.load(f)
    if chat_template is not None:
        jailbreak_prompts = [chat_template.format(each["corrupted_question"]) for each in test_pairs]
    else:
        jailbreak_prompts = [each["corrupted_question"] for each in test_pairs]
    return jailbreak_prompts

def prepare_format_dataset(
    path: str = "google/IFEval",
    chat_template: str = None,
):
    from datasets import load_dataset
    dataset = load_dataset(path, split="train")
    dataset = dataset["prompt"]
    if chat_template is not None:
        dataset = [chat_template.format(each) for each in dataset]

    return dataset


def prepare_cad_dataset(
    path: str = "dataset/test.json",
    chat_template: str = None,
):
    positive_list = []
    negative_list = []

    with open(path, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter='\t')
        for row in csv_reader:
            if row['Sentiment'] == 'Positive':
                positive_list.append(row['Text'])
            elif row['Sentiment'] == 'Negative':
                negative_list.append(row['Text'])
    
    if chat_template is not None:
        positive_list = [chat_template.format(each) for each in positive_list]
        negative_list = [chat_template.format(each) for each in negative_list]

    return positive_list, negative_list