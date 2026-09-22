import os
import argparse

PARSER = argparse.ArgumentParser(description="UNet-3D")

PARSER.add_argument('--data_dir', dest='data_dir', required=True)
PARSER.add_argument('--raw_dir', dest='raw_dir', required = True)
PARSER.add_argument('--log_dir', dest='log_dir', type=str, default="/tmp")
PARSER.add_argument('--image_spacings', dest='image_spacing_dic', type=dict, default=dict())
PARSER.add_argument('--save_ckpt_path', dest='save_ckpt_path', type=str, default="")
PARSER.add_argument('--load_ckpt_path', dest='load_ckpt_path', type=str, default="")
PARSER.add_argument('--loader', dest='loader', default="pytorch", type=str)
# PARSER.add_argument("--local_rank", default=os.environ.get("LOCAL_RANK", 0), type=int)
PARSER.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)))

PARSER.add_argument("--disable_logging", action="store_true")  # Add the --disable_logging argument
PARSER.add_argument('--epochs', dest='epochs', type=int, default=1)
PARSER.add_argument('--quality_threshold', dest='quality_threshold', type=float, default=0.908)
PARSER.add_argument('--ga_steps', dest='ga_steps', type=int, default=1)
PARSER.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=4)
PARSER.add_argument('--batch_size', dest='batch_size', type=int, default=2)
PARSER.add_argument('--layout', dest='layout', type=str, choices=['NCDHW'], default='NCDHW')
PARSER.add_argument('--input_shape', nargs='+', type=int, default=[128, 128, 128])
PARSER.add_argument('--val_input_shape', nargs='+', type=int, default=[128, 128, 128])
PARSER.add_argument('--seed', dest='seed', default=-1, type=int)
PARSER.add_argument('--num_workers', dest='num_workers', type=int, default=2)
PARSER.add_argument('--exec_mode', dest='exec_mode', choices=['train', 'evaluate'], default='train')
#PARSER.add_argument('--results_dir', dest='results_dir', required=True)

PARSER.add_argument('--benchmark', dest='benchmark', action='store_true', default=False)
PARSER.add_argument('--amp', dest='amp', action='store_true', default=False)
PARSER.add_argument('--optimizer', dest='optimizer', default="sgd", choices=["sgd", "adam", "lamb"], type=str)
PARSER.add_argument('--learning_rate', dest='learning_rate', type=float, default=1.0)
PARSER.add_argument('--init_learning_rate', dest='init_learning_rate', type=float, default=1e-4)
PARSER.add_argument('--lr_warmup_epochs', dest='lr_warmup_epochs', type=int, default=0)
PARSER.add_argument('--lr_decay_epochs', nargs='+', type=int, default=[])
PARSER.add_argument('--lr_decay_factor', dest='lr_decay_factor', type=float, default=1.0)
PARSER.add_argument('--lamb_betas', nargs='+', type=int, default=[0.9, 0.999])
PARSER.add_argument('--momentum', dest='momentum', type=float, default=0.9)
PARSER.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0)
PARSER.add_argument('--evaluate_every', '--eval_every', dest='evaluate_every', type=int, default=None)
PARSER.add_argument('--start_eval_at', dest='start_eval_at', type=int, default=None)
PARSER.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False)
PARSER.add_argument('--normalization', dest='normalization', type=str,
                    choices=['instancenorm', 'batchnorm'], default='instancenorm')
PARSER.add_argument('--activation', dest='activation', type=str, choices=['relu', 'leaky_relu'], default='relu')

# ---------------------------------------------------------------------------
# Minato: warm-up calibration of t_out + resume-from-index migration
# ---------------------------------------------------------------------------
def _bool_flag(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


PARSER.add_argument('--calibrate', dest='calibrate', type=_bool_flag, default=True,
                    help='Run the warm-up profiling pass that derives t_out from the '
                         'observed latency distribution. Set to false to use '
                         '--sample_timeout as a fixed threshold.')
PARSER.add_argument('--calib_seconds', dest='calib_seconds', type=float, default=600.0,
                    help='Wall-clock budget for the warm-up profiling pass (default 10 min).')
PARSER.add_argument('--calib_max_samples', dest='calib_max_samples', type=int, default=256,
                    help='Stop profiling once this many samples have been timed.')
PARSER.add_argument('--calib_min_samples', dest='calib_min_samples', type=int, default=24,
                    help='Do not stop on the time budget before this many samples.')
PARSER.add_argument('--calib_warmup_samples', dest='calib_warmup_samples', type=int, default=3,
                    help='Discard the first N timings (cold page cache / allocator).')
PARSER.add_argument('--calib_percentile', dest='calib_percentile', type=float, default=75.0,
                    help='Percentile of the profiled distribution used as t_out.')
PARSER.add_argument('--calib_margin', dest='calib_margin', type=float, default=1.0,
                    help='Multiplier applied to the calibrated percentile. Raise it '
                         '(e.g. 1.2) when the latency distribution is bimodal and P75 '
                         'lands on the edge of the fast mode.')
PARSER.add_argument('--calib_cache', dest='calib_cache', type=str, default="",
                    help='JSON file used to persist/reuse a calibration result.')
PARSER.add_argument('--recalibrate', dest='recalibrate', action='store_true', default=False,
                    help='Ignore --calib_cache and profile again.')
PARSER.add_argument('--t_out_min', dest='t_out_min', type=float, default=0.05)
PARSER.add_argument('--t_out_max', dest='t_out_max', type=float, default=120.0)
PARSER.add_argument('--sample_timeout', dest='sample_timeout', type=float, default=2.0,
                    help='Fallback t_out when calibration is disabled or fails.')

PARSER.add_argument('--checkpoint_copy', dest='checkpoint_copy', type=_bool_flag, default=True,
                    help='Copy the sample at every stage boundary so a migrated '
                         'sample can be resumed safely. Disable only to measure '
                         'the copy overhead.')
PARSER.add_argument('--front_threads', dest='front_threads', type=int, default=2,
                    help='Threads in the front worker pool that execute the pipeline.')
PARSER.add_argument('--num_slow_workers', dest='num_slow_workers', type=int, default=3,
                    help='Background processes that resume migrated samples.')
PARSER.add_argument('--queue_wait_timeout', dest='queue_wait_timeout', type=float, default=30.0)

PARSER.add_argument('--adapt', dest='adapt', action='store_true', default=False,
                    help='Enable the background profiler that rewrites t_out during '
                         'training (paper 4.2). Off by default: with it on, t_out is '
                         'no longer a fixed experimental variable.')
PARSER.add_argument('--adapt_fallback_percentile', dest='adapt_fallback_percentile',
                    type=float, default=90.0,
                    help='Percentile used once too many samples are being migrated.')
PARSER.add_argument('--adapt_window', dest='adapt_window', type=int, default=512)
PARSER.add_argument('--adapt_interval_s', dest='adapt_interval_s', type=float, default=5.0)
PARSER.add_argument('--adapt_min_samples', dest='adapt_min_samples', type=int, default=64)
PARSER.add_argument('--adapt_max_step_ratio', dest='adapt_max_step_ratio',
                    type=float, default=2.0,
                    help='t_out may not move by more than this factor per update.')
PARSER.add_argument('--adapt_escalate_rate', dest='adapt_escalate_rate',
                    type=float, default=0.35,
                    help='Migration rate above which the percentile escalates.')
PARSER.add_argument('--adapt_deescalate_rate', dest='adapt_deescalate_rate',
                    type=float, default=0.20)
PARSER.add_argument('--adapt_cooldown_s', dest='adapt_cooldown_s', type=float, default=20.0)
PARSER.add_argument('--adapt_censoring', dest='adapt_censoring', type=str,
                    default='reconstructed', choices=['reconstructed', 'censored'],
                    help="'censored' feeds a migrated sample's cut-off time back into "
                         'the percentile, which pins t_out and prevents it from rising.')
PARSER.add_argument('--profile_dir', dest='profile_dir', type=str, default="",
                    help='Directory for calibration JSON and per-process event CSVs.')
PARSER.add_argument('--profile_fast_samples', dest='profile_fast_samples',
                    action='store_true', default=False,
                    help='Also record per-stage timings for non-migrated samples.')
PARSER.add_argument('--minato_verbose', dest='verbose_minato', action='store_true', default=False)

PARSER.add_argument('--oversampling', dest='oversampling', type=float, default=0.4)
PARSER.add_argument('--overlap', dest='overlap', type=float, default=0.5)
PARSER.add_argument('--include_background', dest='include_background', action='store_true', default=False)
PARSER.add_argument('--cudnn_benchmark', dest='cudnn_benchmark', action='store_true', default=False)
PARSER.add_argument('--cudnn_deterministic', dest='cudnn_deterministic', action='store_true', default=False)
