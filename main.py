import datetime
from pathlib import Path

import whynot as wn
import mlflow

from causal_rl import ppo, vpg
from causal_rl.common import compute_causal_factor1, plot_agent_behaviors
from causal_rl.common import NoTreatmentPolicy, RandomPolicy, MaxTreatmentPolicy

# Initialize mlflow tracking
mlflow.set_tracking_uri("file:/path/to/your/mlflow/tracking")  # Replace with your desired path
mlflow.set_experiment("causal_rl_experiments")

env = wn.gym.make("HIV-v0")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
checkpoints_path = log_dir.joinpath("checkpoints")
checkpoints_path.mkdir(parents=True, exist_ok=True)

EPOCHS = 20
EPISODES_PER_EPOCH = 16

# Define a function to log parameters and metrics with mlflow
def log_to_mlflow(run_name, params, metrics):
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

# Training and logging with mlflow for VPG
def train_and_log_vpg(model_name, model_class):
    model = model_class(env)
    model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_save_path = checkpoints_path.joinpath(f"{model_name}_{datetime.datetime.now():%d%m%y%H%M%S}.pt")
    model.save(model_save_path)
    mlflow_run_name = f"{model_name}_experiment_{datetime.datetime.now():%d%m%y%H%M%S}"
    log_to_mlflow(mlflow_run_name, {}, {})  # Modify params and metrics as needed

# Training and logging with mlflow for CausalPG
def train_and_log_causal_pg(model_name, model_class):
    model = model_class(compute_causal_factor1, env)
    model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_save_path = checkpoints_path.joinpath(f"{model_name}_{datetime.datetime.now():%d%m%y%H%M%S}.pt")
    model.save(model_save_path)
    mlflow_run_name = f"{model_name}_experiment_{datetime.datetime.now():%d%m%y%H%M%S}"
    log_to_mlflow(mlflow_run_name, {}, {})  # Modify params and metrics as needed

# Training and logging with mlflow for PPO
def train_and_log_ppo(model_name, model_class):
    model = model_class(env)
    model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_save_path = checkpoints_path.joinpath(f"{model_name}_{datetime.datetime.now():%d%m%y%H%M%S}.pt")
    model.save(model_save_path)
    mlflow_run_name = f"{model_name}_experiment_{datetime.datetime.now():%d%m%y%H%M%S}"
    log_to_mlflow(mlflow_run_name, {}, {})  # Modify params and metrics as needed

# Training and logging with mlflow for CausalPPO
def train_and_log_causal_ppo(model_name, model_class):
    model = model_class(compute_causal_factor1, env)
    model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_save_path = checkpoints_path.joinpath(f"{model_name}_{datetime.datetime.now():%d%m%y%H%M%S}.pt")
    model.save(model_save_path)
    mlflow_run_name = f"{model_name}_experiment_{datetime.datetime.now():%d%m%y%H%M%S}"
    log_to_mlflow(mlflow_run_name, {}, {})  # Modify params and metrics as needed

# Training and logging for each model
train_and_log_vpg("vpg_model", vpg.VPG)
train_and_log_causal_pg("causal_pg_model", vpg.CausalPG)
train_and_log_ppo("ppo_model", ppo.PPO)
train_and_log_causal_ppo("causal_ppo_model", ppo.CausalPPO)

# Load models if needed
vpg_model.load("logs/checkpoints/vpg_model_*.pt")
causal_pg_model.load("logs/checkpoints/causal_pg_model_*.pt")
ppo_model.load("logs/checkpoints/ppo_model_*.pt")
causal_ppo_model.load("logs/checkpoints/causal_ppo_model_*.pt")

# Plot behaviors
agents = {
    "vpg": vpg_model,
    "causal_pg": causal_pg_model,
    "ppo": ppo_model,
    "causal_ppo": causal_ppo_model,
}

for name, agent in agents.items():
    plot_agent_behaviors(
        {name: agent},
        env,
        state_names=wn.hiv.State.variable_names(),
        max_timesteps=100,
        save_path=Path("logs").joinpath(f"{name}_alone_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"),
        show_plot=False,
    )
    plot_agent_behaviors(
        {
            "random": RandomPolicy(),
            "max": MaxTreatmentPolicy(),
            "none": NoTreatmentPolicy(),
            name: agent,
        },
        env,
        state_names=wn.hiv.State.variable_names(),
        max_timesteps=100,
        save_path=Path("logs").joinpath(f"{name}_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"),
        show_plot=False,
    )

plot_agent_behaviors(
    {
        "random": RandomPolicy(),
        "max": MaxTreatmentPolicy(),
        "none": NoTreatmentPolicy(),
        "vpg": vpg_model,
        "causal_pg": causal_pg_model,
        "ppo": ppo_model,
        "causal_ppo": causal_ppo_model,
    },
    env,
    state_names=wn.hiv.State.variable_names(),
    max_timesteps=100,
    save_path=Path("logs").joinpath(f"all_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"),
    show_plot=False,
)
