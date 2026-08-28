"""Single-run explore -> behavioral-clone -> finetune migration.

One config, one budget split into an IntrinsicPPO exploration phase and a MORLAX
finetuning phase, bridged by behavioral cloning of the exploration Pareto
archive. See moplayground.learning.migration for the pipeline.

Usage:
    python -m scripts.train_migration config/morlax/mohopper_sparse_migration.yaml
    SMORL_RUN_NAME=migrate-thr50-seed0 python -m scripts.train_migration <config>
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os

import moplayground as mop
import minimal_mjx as mm

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help='Path to migration YAML config')
parser.add_argument('--name', type=str, default=None, help='Override config.name')
args = parser.parse_args()

config = mop.utils.read_config(args.env)

run_name = args.name or os.environ.get('SMORL_RUN_NAME')
if run_name:
    config.name = run_name
    print(f'Using run name override: {run_name}')

print('Training', args.env)
print(f"Results dir: {config['save_dir']}/{config['name']}")
env, _ = mop.envs.create_environment(config, for_training=True)
eval_env, _ = mop.envs.create_environment(config, for_training=True)

name = config['save_dir'] + '/' + config['name']
run = mm.utils.logging.initialize_wandb(
    name=name.replace('/', ''),
    entity='raghavthakar-oregon-state-university',
    project='SMORL',
    config=dict(config),
)
mop.train_migration(config, env, eval_env, run)
