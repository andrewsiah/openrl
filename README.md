Setup UV:
```
uv sync
```

To run:
```
cd src
accelerate launch --config_file accelerate_config.yaml train.py
```

Configs are in `src/config.yaml` and `src/accelerate_config.yaml`.


TODO:
- [ ] We can't add more wandb metrics for now. Fix this
- [ ] Add more evals, look into evalchemy
- [ ] Add sweep code
- [ ] Add more metrics that we're interested in
