from analysis import analysis_reconstruction as recon
import argparse
from glob import glob
import os

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--results', nargs='+', required=True, help='list of result files')
    a.add_argument('--gt_path', type=str, required=True, help='path to ground truth folder')
    a.add_argument('--output_dir', type=str, required=True, help='path to output folder')

    opt = a.parse_args()

    # read result files
    assert len(opt.results) == 3, 'Wrong number of result files.'
    data_file_static, data_file_static_dynamic, data_file_static_dynamic_no_sync = opt.results

    # read gt control points
    gt_static_file = sorted(glob(os.path.join(opt.gt_path, '*.txt')))
    assert len(gt_static_file) >= 2, 'Not enough control point files found.'

    # specify output directory
    output_dir = opt.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    recon.main(data_file_static, data_file_static_dynamic, data_file_static_dynamic_no_sync, gt_static_file, output_dir)