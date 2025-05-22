import argparse
from loguru import logger
from src.datasets.xinhua import get_task_datasets
from evaluator import BaseEvaluator
# from src.llms import GPT
from src.llms import Qwen_7B_Chat
from src.llms import DeepSeek_R1_Distill_Qwen_7B_Chat
from src.llms import Qwen2_5_7B_Chat
from src.tasks.summary import Summary
from src.tasks.continue_writing import ContinueWriting
from src.tasks.hallucinated_modified import HalluModified
from src.tasks.quest_answer import QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs


parser = argparse.ArgumentParser()

# Dataset related options
parser.add_argument('--data_path', default='data/crud_split/split_merged.json', help="Path to the dataset")
parser.add_argument('--shuffle', type=bool, default=True, help="Whether to shuffle the dataset")

# Metric related options
parser.add_argument('--quest_eval', action='store_true', help="Whether to use QA metrics(RAGQuestEval)")
parser.add_argument('--bert_score_eval', action='store_true', help="Whether to use bert_score metrics")

# Evaluation related options
parser.add_argument('--task', default='event_summary', help="Task to perform")
parser.add_argument('--num_threads', type=int, default=1, help="Number of threads")
parser.add_argument('--show_progress_bar', action='store', default=True, type=bool, help="Whether to show a progress bar")
parser.add_argument('--contain_original_data', action='store_true', help="Whether to contain original data")

args = parser.parse_args()
logger.info(args)

task_mapping = {
    'event_summary':[Summary],
    'continuing_writing': [ContinueWriting],
    'hallu_modified': [HalluModified],
    # 'quest_answer': [QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs],
    'quest_answer': [QuestAnswer1Doc],
    'all': [Summary, ContinueWriting, HalluModified, QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs]
}

if args.task not in task_mapping:
    raise ValueError(f"Unknown task: {args.task}")

tasks = [task(use_quest_eval=args.quest_eval, use_bert_score=args.bert_score_eval) for task in task_mapping[args.task]]

datasets = get_task_datasets(args.data_path, args.task)

for task, dataset in zip(tasks, datasets):
    evaluator = BaseEvaluator(task, dataset, num_threads=args.num_threads)
    evaluator.run(show_progress_bar=args.show_progress_bar, contain_original_data=args.contain_original_data)

