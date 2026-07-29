import matplotlib
matplotlib.use('Agg')
import os
import moplayground as mop
import minimal_mjx as mm
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Path to training YAML config")
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help=(
        "Override config.name (output subdir under save_dir). "
        "Also accepted via env SMORL_RUN_NAME. Use a new name each "
        "run to avoid overwriting previous results."
    ),
)
args = parser.parse_args()
TRAIN_KWARGS = {}
EVAL_KWARGS  = {}

# Read in configs
train_config = mop.utils.read_config(args.env)
eval_config  = mop.utils.read_config(args.env)

# Optional run-name override so repeats land next to (not on top of) prior runs.
run_name = args.name or os.environ.get('SMORL_RUN_NAME')
if run_name:
    train_config.name = run_name
    eval_config.name = run_name
    print(f'Using run name override: {run_name}')

# Environment-specific config handling...
match train_config.env:
    case 'BRUCE':
        EVAL_KWARGS = {'manual_speed': [0.0, 0.0, 0.0], 'idealistic': True}

# Create environments
print('Training', args.env)
print(f"Results dir: {train_config['save_dir']}/{train_config['name']}")
env, env_cfg = mop.envs.create_environment(train_config, for_training=True, **TRAIN_KWARGS)
eval_env, _  = mop.envs.create_environment(eval_config, for_training=True, **EVAL_KWARGS)

name = train_config['save_dir'] + '/' + train_config['name']
run = mm.utils.logging.initialize_wandb(
    name    = name.replace('/', ''),
    entity  = 'raghavthakar-oregon-state-university',
    project = 'SMORL',
    config  = dict(train_config)
)
mop.learning.train_policy(train_config, env, eval_env, run)
