If you have UV installed:
```
uv sync
```

To run:
```
cd src
accelerate launch --config_file accelerate_config.yaml train.py
```

Configs are in `src/config.yaml` and `src/accelerate_config.yaml`.

Sweep in `sweep.py`.
```
uv run sweep.py
```


TODO:
- [ ] Add more evals, look into evalchemy
- [ ] Add more metrics that we're interested in
- [ ] Train on other datasets [MATH]
- [ ] Make training faster, eval on vllm and on all N gpus instead of main and transformers
- [x] We can't add more wandb metrics for now. Fix this
- [x] Add sweep code


Baselines:

[Qwen 2.5 Blog](https://qwenlm.github.io/blog/qwen2.5-llm/)

Resources:

[Anton's Replication Github Gist](https://t.co/VnhlhkfI4r)

[Will's Replication Github Gist](https://t.co/5qoV7Ul0rR)




Cite:

```
@misc{openrl2025,
      title={OpenRL: Replicating RL on LLMs ala Deepseek's R1 and OpenAI's O-Series}, 
      author={Andrew Wei Tung Siah},
      year={2025},
      url={https://github.com/andrewsiah/openrl}
}
```