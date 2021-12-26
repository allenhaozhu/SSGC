# TextSSGC

Implementation of SSGC for text classification following
([Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) and [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)).
Code modified from the original repo of
SGC(https://github.com/Tiiiger/SGC).

For reference, 

Dataset | Accuracy 
:------:|:------:|
20NG    | 88.6 %         
R8      | 97.4 %          
R52     | 94.5 %          
Ohsumed | 68.5 %         
MR      | 76.7 % 
## Data

We get the data from [original repo](https://github.com/yao8839836/text_gcn) and
process with `remove_words.py`, `build_graph.py`. We modify these two files from
the original repo and made some slight changes.
We share the processed data in a zip file
[online](https://drive.google.com/file/d/10kx3z3bjYFoeRjjg1_DZOAP39Jln0BCh/view?usp=sharing).
Please uncompress the file and put it under `data/`.
If you want to redo the processing, see options with `python build_graph.py
--help` and `python remove_words.py --help`.

Training with TextGCN and TextSGC can be memory intensive (10+ GB CPU memory
required for the 20ng dataset). Therefore, we share a preprocessed version of
the data [online](https://drive.google.com/drive/u/2/my-drive). To use it,
download and decompress to `./preprocessed`, and use the `--preprocessed` flag.

## Usage

There is only one hyperparameter needs to be set and we tune it with hyperopt.
We provide the tuned hyperparameters under `tuned_result`.
You can replicate the paper reported result by
```
python train.py --dataset <dt> --tuned
```

See `tuning.py` for the tuning details.
