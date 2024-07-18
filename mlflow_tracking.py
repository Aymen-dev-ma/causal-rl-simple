import datetime
from pathlib import Path

import mlflow
from mlflow import log_metric, log_param
import whynot as wn

from causal_rl import ppo, vpg  # noqa: F401
from causal_rl.common import compute_causal_factor1, plot_agent_behaviors
from causal_rl.common import NoTreatmentPolicy, RandomPolicy, MaxTreatmentPolicy

# Set the experiment name for mlflow
experiment_name = "causal_rl_experiment"
mlflow.set_experiment(experiment_name)

# Create an mlflow run to track this script's execution
with mlflow.start_run():
    # Log parameters relevant to the experiment
    log_param("epochs", 20)
    log_param("episodes_per_epoch", 16)

    # Define the directory paths for logs and checkpoints
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    checkpoints_path = log_dir.joinpath("checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    # Training and saving VPG model
    vpg_model = vpg.VPG(env)
    vpg_model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_name = f"vpg_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
    vpg_model.save(checkpoints_path.joinpath(model_name))
    mlflow.log_artifact(str(checkpoints_path.joinpath(model_name)))

    # Training and saving CausalPG model
    causal_pg_model = vpg.CausalPG(compute_causal_factor1, env)
    causal_pg_model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_name = f"causal_pg_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
    causal_pg_model.save(checkpoints_path.joinpath(model_name))
    mlflow.log_artifact(str(checkpoints_path.joinpath(model_name)))

    # Training and saving PPO model
    ppo_model = ppo.PPO(env)
    ppo_model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_name = f"ppo_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
    ppo_model.save(checkpoints_path.joinpath(model_name))
    mlflow.log_artifact(str(checkpoints_path.joinpath(model_name)))

    # Training and saving CausalPPO model
    causal_ppo_model = ppo.CausalPPO(compute_causal_factor1, env)
    causal_ppo_model.train(
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        log_dir=log_dir,
        PLOT_REWARDS=True,
        VERBOSE=True,
        TENSORBOARD_LOG=True,
        SHOW_PLOTS=False,
    )
    model_name = f"causal_ppo_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
    causal_ppo_model.save(checkpoints_path.joinpath(model_name))
    mlflow.log_artifact(str(checkpoints_path.joinpath(model_name)))

    # Load models for evaluation
    vpg_model.load("logs/12072020180721/VPG_HIV-v0.pt")
    causal_pg_model.load("logs/12072020190532/CausalPG_HIV-v0.pt")
    ppo_model.load("logs/12072020182111/PPO_HIV-v0.pt")
    causal_ppo_model.load("logs/12072020183315/CausalPPO_HIV-v0.pt")

    agents = {
        "vpg": vpg_model,
        "causal_pg": causal_pg_model,
        "ppo": ppo_model,
        "causal_ppo": causal_ppo_model,
    }

    # Evaluate and plot agent behaviors
    for name, a in agents.items():
        plot_agent_behaviors(
            {name: a},
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
                name: a,
            },
            env,
            state_names=wn.hiv.State.variable_names(),
            max_timesteps=100,
            save_path=Path("logs").joinpath(
                f"{name}_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
            ),
            show_plot=False,
        )

    # Evaluate and plot all agents' behaviors together
    plot_agent_behaviors(
        {
            "random": RandomPolicy(),
            "max": MaxTreatmentPolicy(),
            "none": NoTreatmentPolicy(),
            **agents,
        },
        env,
        state_names=wn.hiv.State.variable_names(),
        max_timesteps=100,
        save_path=Path("logs").joinpath(
            f"all_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
        ),
        show_plot=False,
    )
    mlflow.log_artifact(str(Path("logs").joinpath("all_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png")))

# End of mlflow run
