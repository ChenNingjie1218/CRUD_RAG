import copy
import json
import os
from abc import ABC
from loguru import logger
from tqdm import tqdm
from threading import Lock
from src.llms.base import BaseLLM
from src.tasks.base import BaseTask
import concurrent.futures
import time

class BaseEvaluator(ABC):
    def __init__(self, task: BaseTask,
        dataset: list[dict], output_dir: str = './output', num_threads: int = 40):
        """
        Args:
            task (BaseTask): The task for evaluation.
            dataset (list[dict]): The dataset for evaluation.
            output_dir (str): The directory for result output and caching.
        """
        self.dataset = dataset
        self.task = task
        self.lock = Lock()
        self.num_threads = num_threads

        # output_dir = os.path.join(output_dir, f'')
        
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        self.output_path = os.path.join(
            output_dir, f'{self.task.__class__.__name__}.json'
        )

    def task_generation(self, data_point):
        return self.task.model_generation(data_point)

    def multithread_batch_scoring(self, dataset: list[dict], sort=True, show_progress_bar=False, contain_original_data=False) -> list[dict]:
        """Perform batch scoring on the given dataset.

        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            list[dict]: List of results.
        """

        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        def process_data_point(data_point):
            if data_point['ID'] in saved_ids:
                return None  # Skip results that have already been evaluated and are valid
            try:
                # start_time = time.time()
                generated_text = self.task_generation(data_point)
                # end_time = time.time()
                # 计算耗时
                # elapsed_time = end_time - start_time
                # logger.info(f"process_data_point检索耗时: {elapsed_time:.2f} 秒")
                # TODO fix bugs
                if generated_text == '","msg":"request openai failed"' or generated_text == '':
                    return None
                
                data_point["generated_text"] = generated_text

                score_dict = self.task.scoring(data_point)
                if score_dict is None:
                    return None
                
                result = {'id': data_point['ID'], **score_dict}
                
                if contain_original_data:
                    result['original_data'] = data_point

                return result
            
            except Exception as e:
                logger.warning(repr(e))
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_results = list(tqdm(executor.map(process_data_point, dataset), total=len(dataset)))
        
        results.extend([result for result in future_results if result is not None])
        
        return sorted(results, key=lambda x: x['id']) if sort else results

    def save_output(self, output: dict) -> None:
        """Save evaluation results."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    
    def read_output(self) -> dict:
        with open(self.output_path) as f:
            return json.load(f)

    def run(self, sort = True, show_progress_bar = False, contain_original_data = True) -> dict:
        """Run a complete evaluation.

        Args:            
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.
            contain_original_data (bool): Whether to include original data in the results for debugging.

        Returns:
            dict: Output dictionary contains fields such as: info, overall, results, etc.
        """
        info = {
            'task': self.task.__class__.__name__, 
            # 'llm': str(self.llm.params),
        }

        results = self.multithread_batch_scoring(self.dataset, sort, show_progress_bar, contain_original_data)
        valid_results = self.remove_invalid(results)

        try:
            overall = self.task.compute_overall(valid_results) if len(valid_results) > 0 else {}\
            # 保存用于评估的RAGQuestEval QA问答对
            if self.task.use_quest_eval:
                self.lock.acquire()
                self.task.quest_eval.save_quest_gt(self.task.__class__.__name__)
                self.lock.release()
        
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()

        self.save_output(output:={'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output

    @staticmethod
    def remove_invalid(results: list[dict]) -> list[dict]:
        """Remove invalid results from the list and return the cleaned results."""
        return [result for result in results if result['valid']]

    def batch_scoring(self, dataset:list[dict], sort = True, show_progress_bar = False, contain_original_data = False):
        """Perform batch scoring on the given dataset.
        
        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.
        
        Returns:
            list[dict]: List of results.
        """
        
        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        for data_point in (tqdm(dataset, desc=self.model.params['model_name']) if show_progress_bar else dataset):
            if data_point['ID'] in saved_ids:
                continue  # Skip results that have already been evaluated and are valid
            try:
                start_time = time.time()
                generated_text = self.task_generation(data_point)
                end_time = time.time()
                # 计算耗时
                elapsed_time = end_time - start_time
                logger.info(f"batch_scoring检索耗时: {elapsed_time:.2f} 秒")
                data_point["generated_text"] = generated_text
                result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                if contain_original_data:
                    result['original_data'] = data_point
                results.append(result)
            except Exception as e:
                logger.warning(repr(e))

        return sorted(results, key=lambda x: x['id']) if sort else results
