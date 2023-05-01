# medical-image-generation

Fine tuning control net for medical chest x ray images

## Training

```
python training.py
```

## Adding ControlNet archirtecture

This file adds control net archirtecture to stable diffusion model

```
python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt
```
## Adding laplace gaussian filter
```
python filter.py
```
