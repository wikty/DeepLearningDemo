# 简介

本项目可以作为 [PyTorch](https://pytorch.org) 深度学习项目的原型和起点，作为原型项目有哪些特色呢？

* 项目模块化
* 模型训练和评估自动化
* 超参数搜索自动化
* 不同的项目遵循相似的流水线

该项目主要参考了 [stanford cs230](https://cs230-stanford.github.io) 课程所开源的项目（在此感谢相关贡献者）。

# 项目介绍

## 任务

该原型项目将围绕[命名实体识别](https://en.wikipedia.org/wiki/Named-entity_recognition)任务来展开。

命名实体识别是指从文本中识别具有特殊意义的实体，如人名、地名、商品名、专有名词等，在信息提取、机器翻译、问答系统等各种自然语言处理系统中都具有重要的作用。从机器学习分类的角度讲，命名实体识别任务属于有监督学习，是一个多分类问题。

## 数据集

数据集采用 Kaggle 中的 [Annotated Corpus for Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/) 中的 [ner_dataset.csv](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner_dataset.csv/4)。

该 CSV 文件，每行是对一个单词的标注，并通过行的顺序和 **Sentence #** 列表示单词和句子的关系。文件一共含有四列：

* **Sentence #**

  句子 ID，值是类似 `Sentence: 1` 或为空，为空则表示跟最近的非空值保持一致。

* **Word**

  单词，拥有同一个句子 ID 的表示这些单词属于同一个句子。

* **POS**

  词性标注，对单词的词性进行标注（词性对命名实体会有影响）。

* **Tag**

  命名实体的标注，对于非实体标注为 `O`，其它实体常用标签有：`B-art`, `B-geo`, `B-org`, `I-geo`, `I-org` 等，分别用来表示不同类型的命名实体。

文件内容示例：

| Sentence #  | Word          | POS  | Tag   |
| ----------- | ------------- | ---- | ----- |
| Sentence: 1 | Thousands     | NNS  | O     |
|             | of            | IN   | O     |
|             | demonstrators | NNS  | O     |
|             | have          | VBP  | O     |
|             | marched       | VBN  | O     |
|             | through       | IN   | O     |
|             | London        | NNP  | B-geo |
|             | to            | TO   | O     |
|             | protest       | VB   | O     |

# 代码结构

```
data/
	ner_dateset.csv
    train/
    val/
    test/
experiments/
	base_model/
		params.json
build_dataset.py
load_dataset.py
model.py
train.py
hyperparams.py
evaluate.py
```

* `data/`，项目数据。数据一般经过处理后分为 `train/`, `val/`, `test/`，它们分别用于模型的训练、超参调优和模型评价、以及模型性能的最终评估。
* `experiments/`，存放模型实验的配置信息和搜索超参的结果。每个实验的配置信息，对应一个子目录，如上例中的 `experiments/base_model/`，配置信息将保存在 `params.json` 文件中。
* `build_dataset.py`，从数据来源创建以及转换数据集，并将数据集分为 train, val 和 test 三部分。
* `load_dataset.py`，加载数据集，并做适当处理，转换为模型可用的数据形式。
* `model.py`，模型的定义。
* `train.py`，用 train 数据集来训练模型，并在过程中使用 val 数据集来评估模型。
* `hyperparams.py`，运行多次 `train.py` 来搜索最优超参。
* `evaluate.py`，用 test 数据集对模型性能进行的最后评估，一般应该在模型训练完成并超参调优后再运行最后的性能评估，以避免在 test 数据集上过拟合。

# 运行项目

大致来看项目的运行分为以下几个阶段：

1. 准备数据：收集数据并将其转换为项目可用的形式。
2. 进行实验：使用训练数据集来进行模型的学习。
3. 超参搜索：用不同超参训练模型来获得最优结果。
4. 模型评价：评价模型在测试数据集上的表现。

## 准备数据

1. 首先从 [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner_dataset.csv/4) 下载数据文件，并将其保存到项目中的 `data/ner_dataset.csv` 位置。
2. 然后运行 `python build_dataset.py  --data-file data/ner_dataset.csv`，来将数据集划分为训练集、验证集以及测试集，并提取单词词汇表和标注词汇表。

## 进行实验

**配置实验参数**

在项目中的 `experiments` 文件夹中创建本次实验的配置信息。

例如创建子目录 `base_model` 并新建文件 `params.json` ，写入超参等配置信息到文件中：`{"num_epochs": 100, "batch_size": 4, "learning_rate": 1e-6}`。

**开始训练模型**

运行 `python train.py --model-dir experiments/base_model/`。

模型训练结束后会在相应的参数配置目录中生成如下实验结果数据：

- `train.log`: 训练期间打印到标准输出中的内容。
- `train_summaries`: train summaries for TensorBoard (TensorFlow only)
- `eval_summaries`: eval summaries for TensorBoard (TensorFlow only)
- `last_weights`: weights saved from the 5 last epochs
- `best_weights`: best weights (based on dev accuracy)

## 超参搜索

以搜索超参 learning rate 为例，首先要创建目录 `experiments/learning_rate/`，并新建文件 `params.json` 以写入参数配置信息。

然后运行： `python hyperparams.py --hyper-dir experiments/learning_rate/`。

## 实验结果

运行 `python expboard.py` 来查看对比多次实验的结果。

It will display a table synthesizing the results like this that is compatible with markdown:

```
|                                               |   accuracy |      loss |
|:----------------------------------------------|-----------:|----------:|
| experiments/base_model                        |   0.989    | 0.0550    |
| experiments/learning_rate/learning_rate_0.01  |   0.939    | 0.0324    |
| experiments/learning_rate/learning_rate_0.001 |   0.979    | 0.0623    |
```

## 模型评价

为了防止在测试集上过拟合，应该只有在模型完全训练好以后，再用测试集来评估模型性能。

运行 `python evaluate.py --model-dir experiments/base_model/`

# 扩展项目

### 构建数据集

`build_dataset.py` 将数据源划分为训练集（train）、验证集（val）和测试集（test）。具体来说该脚本主要做了以下工作：

1. 从配置文件和命令行加载解析相关的数据集构建参数。可以通过 `lib.dataset.builder.get_parser()` 来得到一个命令行解析器，然后再将项目相关的参数添加进去。
2. 载入数据源，并将数据源划分为训练集、验证集和测试集。可以通过继承 `lib.dataset.builder.BaseBuilder` 并实现方法 `load` 和 `dump` 来完成，其中 `load` 用于加载数据源，`dump` 用于将数据源导出为三个数据集。

**要点**

- 保证各个数据集分布一致

  训练集上学习到的模型能够具有较好泛化能力的一个重要前提条件是：训练集的数据分布跟真实的数据分布尽可能的一致。因此划分数据源之前，务必要对其进行随机打乱（shuffle），以保证各个数据集分布是一致的。

- 懒惰载入数据

  在某些项目中数据源可能特别大，无法一次载入内存中，但同时我们又想要将其随机打乱，该怎么办？建议在 `load` 时只是加载数据的索引以及存储位置，然后在 `dump` 时再读取数据并划分到相应的数据集中。

- 构建小数据集

  如果想要在正式训练模型之前，先使用小批量的数据来验证模型的可行性，可以通过为 `BaseBuilder` 的参数 `data_factor` 来指定希望构建数据源的比例。

**API**

- `lib.dataset.builder.get_parse()` 
- `lib.dataset.builder.BaseBuilder.load()` 中应该调用 `self.add_sample()` 来添加样本
- `lib.dataset.builder.BaseBuilder.dump()` 中应该调用 `self.builder()` 来得到数据集对象

### 加载数据集

支持返回数据集迭代器和批迭代器

从磁盘载入数据集，需要实现一个 Handler，每次读入一个样本

利用 SGD 训练模型时，一般会对数据集迭代多次，为了更好的避免陷入局部极小值，应该对每次迭代的数据集进行随机打乱。

**API**

`lib.dataset.loader.DatasetHandler`

需要根据项目实现自己的 Loader 类。从磁盘载入数据集，需要实现一个 Handler，每次读入一个样本。为了支持大数据集的读取，应该将数据集随机打乱的逻辑放在此处。

`lib.dataset.loader.DatasetIterator`

一般情况下，不需要重写此类。支持以样本（sample）为单位对数据集进行迭代。虽然该类提供了 `shuffle` 以支持乱序访问样本，但不适用于大数据集，更加通过 Handler 来实现乱序访问的逻辑。

`lib.dataset.loader.BatchIterator`

一般情况下，不需要重写此类。支持以批（batch）为单位对数据集进行迭代。

`lib.dataset.loader.BaseLoader`

需要根据项目实现自己的 Loader 类。需要为该类提供 Handler 和 Transform，然后可以得到 DatasetIterator 或 BatchIterator。



### 定义模型

`model.py` 至少应该定义模型并实现函数 `model_factory(params)`。

* 模型的定义跟具体应用相关

* `model_factory(params)` 应该返回 `(model, optimizer, criterion, metrics)`，依次表示模型实例、优化器、损失函数、模型评价方法字典



### 训练模型



### 超参调优



### 模型比较



