

import sys
import os

sys.path.append('../')

from ULP_ import ULP
from ULP_.ULP_benchmark import benchmark_settings
from ULP_.common import datasets, common_args
from ULP_.exec_main import evaluator, prepare_results

input_dir = './input/'  # The input directory of log file
output_dir = './output'  # The output directory of parsing result

if __name__ == "__main__":

    args = common_args()
    print(args)
    # prepare result_file
    result_file = prepare_results(
        output_dir=output_dir
        
    )
    if args.dataset:
        setting = benchmark_settings[args.dataset]
    else:
        setting = benchmark_settings[datasets[0]]
        
    log_file = setting['log_file']
    indir= input_dir

    # run evaluator for a dataset
    evaluator(
            dataset=args.dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=ULP.LogParser,
            param_dict={
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex']
            },
            result_file="summary_ulp"
    )  # it internally saves the results into a summary file
