# GloVe Word Embedding on screen2words Dataset

## 1 - 数据

未上传所需的数据。[screen2words 数据集](https://github.com/google-research-datasets/screen2words) 可以在 Google 提供的 Github 仓库获取；[预训练的 GloVe word vector](https://nlp.stanford.edu/projects/glove/) 可以在 GloVe 网站获取。

> 在使用中发现 screen2words 数据集中存在一些错误拼写，使用前进行了一些修正。对于难以辨认的错误，忽略不予处理，以零向量代替。

## 2 - 目录结构

> 保存数据的目录未上传，对应目录可以在代码中修改为合适的路径

* 预训练的词向量文件保存在 `glove_data` 文件夹中
* 加载后的 word2vec 模型保存在 `result` 文件夹中
* screen_summary 的 csv 文件保存在 `screen_summaries` 文件夹中

## 3 - 代码文件

### convert.py

读取预训练的词向量并保存模型，加载此模型，依次处理 screen summary 中的单词，替换为对应的词向量表示，最终结果以 `dataframe` 的形式保存在 pkl 文件中，以便后续模型读取

> 可能需要数分钟

### test_pickle.py

读取保存的 pkl 文件，尝试转化为 `torch.tensor`，作为模型中进行调用的参考