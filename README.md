# LightLX

```
  _    _       _     _   _    __  __
 | |  (_) __ _| |__ | |_| |  \ \/ /   LightLX
 | |  | |/ _` | '_ \| __| |   \  /    run models too big for memory
 | |__| | (_| | | | | |_| |___/  \    (and the ones that fit, fast)
 |_____|_|\__, |_| |_|\__|_____/_/\_\
          |___/
```

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

**Run models too big for your RAM on Windows — any Hugging Face model that
fits on disk but not in memory — by streaming weights layer-by-layer.** Point it
at a model directory and go.

READ THE ORIGINAL REPO
But this is for WINDOWS

Step By Step

Step 1 — Set up a project folder

Create a folder for LightLX and drop all the .py files in it:

lightlx/
├── __init__.py
├── __main__.py
├── model.py
├── generic.py      
├── weights.py
├── cache.py
├── cli.py         
├── state.py
└── models/         ← your downloaded models go here

Step 2 — Download a model From HuggingFace

Run this from inside your lightlx/ folder (or anywhere): In CMD
hf download ai name --local-dir ./models/ai name

Step 3

Type in lightlx into CMD

## License

MIT — see [LICENSE](LICENSE). Built on [MLX](https://github.com/ml-explore/mlx)
and [mlx-lm](https://github.com/ml-explore/mlx-lm). Models you run carry their own
licenses.
