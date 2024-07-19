import datetime
from pathlib import Path
import mlflow
import mlflow.pytorch
import whynot as wn

from causal_rl import ppo, vpg  # noqa: F401
from causal_rl.common import compute_causal_factor1, plot_agent_behaviors
from causal_rl.common import NoTreatmentPolicy, RandomPolicy, MaxTreatmentPolicy

env = wn.gym.make("HIV-v0")

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
checkpoints_path = log_dir.joinpath("checkpoints")
checkpoints_path.mkdir(parents=True, exist_ok=True)

EPOCHS = 20
EPISODES_PER_EPOCH = 16

# Set up MLflow experiment
mlflow.set_experiment("HIV-Treatment-Experiment")

def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("episodes_per_epoch", EPISODES_PER_EPOCH)

        model.train(
            epochs=EPOCHS,
            episodes_per_epoch=EPISODES_PER_EPOCH,
            log_dir=log_dir,
            PLOT_REWARDS=True,
            VERBOSE=True,
            TENSORBOARD_LOG=True,
            SHOW_PLOTS=False,
        )

        model_path = checkpoints_path.joinpath(f"{model_name}_{datetime.datetime.now():%d%m%y%H%M%S}.pt")
        model.save(model_path)

        # Log model
        mlflow.pytorch.log_model(model, model_name)
        mlflow.log_artifact(model_path)

vpg_model = vpg.VPG(env)
train_and_log_model(vpg_model, "VPG")

causal_pg_model = vpg.CausalPG(compute_causal_factor1, env)
train_and_log_model(causal_pg_model, "CausalPG")

ppo_model = ppo.PPO(env)
train_and_log_model(ppo_model, "PPO")

causal_ppo_model = ppo.CausalPPO(compute_causal_factor1, env)
train_and_log_model(causal_ppo_model, "CausalPPO")

# Load models (ensure paths are correct)
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
    save_path=Path("logs").joinpath(
        f"all_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
    ),
    show_plot=False,
)
