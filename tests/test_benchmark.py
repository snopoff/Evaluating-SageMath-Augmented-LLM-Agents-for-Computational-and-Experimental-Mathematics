"""Offline integration checks for the published prediction-scoring pipeline."""
import json
from pathlib import Path
from unittest.mock import patch
import hydra.utils as hu
import pytest
from src.benchmark.benchmark import Benchmark
from src.benchmark.components.config import BenchmarkConfig

@pytest.mark.parametrize('prediction,reference', [('2+2','4'), (['1+n','n-1'],['n+1','n-1']), ('Tuple(1+n,2*n)','Tuple(n+1,n+n)')])
def test_saved_prediction_scoring_and_hydra(tmp_path, prediction, reference):
    path=tmp_path/'predictions.json'
    (tmp_path/'missing_tags.json').write_text('[]')
    path.write_text(json.dumps([{'id':'test-item','question':'A test question','model_sympy_answer':prediction,'ground_truth_sympy_answer':reference}]))
    cfg={'_target_':'src.benchmark.benchmark.Benchmark','config':{'_target_':'src.benchmark.components.config.BenchmarkConfig','predictions_path':str(path),'output_dir':str(tmp_path/'scored'),'tags_path':str(tmp_path/'missing_tags.json'),'answer_types_path':None,'progress_logs':False,'print_statistics':False}}
    # A symbolic accept must never instantiate a paid judge.
    with patch('src.benchmark.components.judge_stage.instantiate_judges',side_effect=AssertionError('Unexpected judge invocation')):
        benchmark=hu.instantiate(cfg);result=benchmark.run()
    assert isinstance(benchmark, Benchmark)
    assert result['rows']==1 and result['final_correct']==1
    assert list((tmp_path/'scored').glob('*final*.json'))

def test_yaml_paths_are_coerced():
    cfg=BenchmarkConfig(predictions_path='example.json',output_dir='generated')
    assert isinstance(cfg.predictions_path,Path) and isinstance(cfg.output_dir,Path)
