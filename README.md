# stable-diffusion-docker

雑に [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) を試すやつ。

## Usage

`./stable-diffusion/models/` に <https://huggingface.co/stabilityai/stable-diffusion-2-1> 等から入手した Model を置く。

```bash
docker compose up -d
docker compose exec stable-diffusion python3.10 scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt ./models/v2-1_768-ema-pruned.ckpt --config ./configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --seed -1 --n_samples 1 --n_iter 1 --device cuda --outdir ./outputs
```
