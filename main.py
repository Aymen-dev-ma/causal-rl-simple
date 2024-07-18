import datetime
from pathlib import Path

import mlflow
from mlflow import log_metric, log_param

import whynot as wn
from causal_rl import ppo, vpg
from causal_rl.common import compute_causal_factor1, plot_agent_behaviors
from causal_rl.common import NoTreatmentPolicy, RandomPolicy, MaxTreatmentPolicy

# Set the tracking URI for mlflow to a local directory where you have write permissions
mlflow.set_tracking_uri("file:/Users/aymennasri/mlflow_tracking")

env = wn.gym.make("HIV-v0")

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
checkpoints_path = log_dir.joinpath("checkpoints")
checkpoints_path.mkdir(parents=True, exist_ok=True)

EPOCHS = 20
EPISODES_PER_EPOCH = 16

# Function to log parameters and metrics to mlflow
def log_to_mlflow(params, metrics):
    for key, value in params.items():
        log_param(key, value)
    for key, value in metrics.items():
        log_metric(key, value)

# Training and logging VPG model
vpg_model = vpg.VPG(env)
for epoch in range(EPOCHS):
    for episode in range(EPISODES_PER_EPOCH):
        # Perform training steps here
        # Replace with actual training logic
        train_stats = {}  # Example statistics; replace with actual values
        log_to_mlflow({"epoch": epoch, "episode": episode}, train_stats)

# Saving VPG model
model_name = f"vpg_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
vpg_model.save(checkpoints_path.joinpath(model_name))

# Repeat similar process for other models (causal_pg, ppo, causal_ppo)...

# Example for plotting agent behaviors
agents = {
    "vpg": vpg_model,
    # Add other models here
}

for name, agent in agents.items():
    plot_agent_behaviors(
        {name: agent},
        env,
        state_names=wn.hiv.State.variable_names(),
        max_timesteps=100,
        save_path=Path("logs").joinpath(
            f"{name}_alone_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
        ),
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
        save_path=Path("logs").joinpath(
            f"{name}_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
        ),
        show_plot=False,
    )

# Example for plotting all agent behaviors together
plot_agent_behaviors(
    {
        "random": RandomPolicy(),
        "max": MaxTreatmentPolicy(),
        "none": NoTreatmentPolicy(),
        **agents,  # Include all trained agents
    },
    env,
    state_names=wn.hiv.State.variable_names(),
    max_timesteps=100,
    save_path=Path("logs").joinpath(
        f"all_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
    ),
    show_plot=False,
)
