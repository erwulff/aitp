# aitp
AI training pipeline using PyTorch (work in progress)

## Supported models

- JEDI-net from [https://github.com/jmduarte/JEDInet-code](https://github.com/jmduarte/JEDInet-code)

## Features

- Running the pipeline's `train.py` script does everything you need; training, validation, plots etc.
- Configure behaviour from a single config file
- Automatically produce various result metrics such as ROC curves, AUC, True Positive & False Positive Rates when training completes
- Automatically save checkpoints when the model improves during training
- Automatically pick the checkpoint with lowest loss before running validation
- Label smoothing


