from submodules.fundamental_metrics.pipeline import FundamentalMetricsPipeline
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent / "submodules" / "EvalMuseMini"))

from submodules.EvalMuseMini.pipeline import EvalMusePipeline
from submodules.QualiClip.pipeline import QualiCLIPPipeline
import hydra
from omegaconf import DictConfig
from utils import extract_column_to_csv


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # extract_column_to_csv(cfg.metadata_path, "img_path", cfg.csv_path)

    # fundamental_pipeline = FundamentalMetricsPipeline(cfg)
    # individual_results = fundamental_pipeline.compute_individual_metrics()
    # grouped_results = fundamental_pipeline.compute_group_metrics()

    # print("Evaluation Results:")
    # for metric, value in grouped_results.items():
    #     print(f"{metric.upper()}: {value:.4f}")

    # eval_muse_pipeline = EvalMusePipeline(cfg)
    # eval_muse_pipeline.evaluate()
    
    # quali_clip_pipeline = QualiCLIPMetric(cfg)
    # quali_clip_pipeline.evaluate()

    # Example configuration (replace with actual Hydra config)
    cfg = DictConfig(
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "evaluation": {
                "qualiclip": {
                    "model_repo": "miccunifi/QualiCLIP",
                    "data_file": "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/metadata.json",
                    "metric": {
                        "_target_": "submodules.QualiClip.metric.QualiCLIPMetric",
                    },
                    "csv_path": "results/results.csv",
                    "save_path": "results/results.json",
                }
            },
        }
    )

    # Initialize and run the pipeline
    pipeline = QualiCLIPPipeline(cfg)
    pipeline.evaluate()

if __name__ == "__main__":
    main()
