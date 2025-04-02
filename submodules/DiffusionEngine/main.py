import os
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
from diffusers import StableDiffusionPipeline
import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    prompts_file = Path(cfg.prompts_file)
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()][:10]

    repo_id = cfg.repo_id
    model_name = cfg.model_name or repo_id.replace('/', '-')
    
    base_dir = Path(hydra.utils.get_original_cwd())
    out_dir = base_dir / "data" / "generated_images" / model_name
    existing_folders = [d for d in out_dir.iterdir() if d.is_dir()]
    folder_number = len(existing_folders)
    
    current_out_dir = out_dir / str(folder_number)
    current_out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    metadata = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt=prompt).images[0]
        img_path = current_out_dir / f"image_{i}.png"
        image.save(img_path)

        metadata.append({
            "image_path": os.path.relpath(img_path, base_dir),
            "prompt": prompt
        })

    metadata_path = current_out_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()