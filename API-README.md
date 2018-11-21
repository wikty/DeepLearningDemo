[TOC]

# lib.dataset

将数据源构建为数据集。

## lib.dataset.builder

### lib.dataset.builder.get_parser

解析数据集构建参数的命令行解析器。

### lib.dataset.builder.Sample

数据样本类。

### lib.dataset.builder.Dataset

数据集类。

支持的属性和方法如下：

* `name` 数据集名字
* `size` 数据集的大小
* `add(sample)` 添加样本到数据集
* 该类还支持迭代访问样本

### lib.dataset.builder.BaseBuilder

#### Description

该模块用于数据集的构建：载入数据源，将其划分为训练集（train）、验证集（val）和测试集（test），并保存到磁盘中。

#### Usage

项目中的数据集构建类应该继承 `lib.dataset.builder.BaseBuilder` 并自定义方法 `load` 和 `dump`。

支持的属性和方法如下：

* `add_sample(**kwargs)` 添加样本到缓存。
* `build(shuffle=True)` 构建并返回一个 tuple `(train_set, val_set, test_set)`，其中的元素是 `lib.dataset.builder.Dataset` 对象。
* `load()` 自定义加载数据源的逻辑。
* `dump()` 自定义导出三个数据集到磁盘的逻辑。

#### Notes

* 加载数据源

  自定义子类需要实现方法 `load` 来加载数据源。每增加一个样本，应该调用 `BaseBuilder.add_sample()` 将其加入到缓存中。

* 导出数据集

  自定义子类需要实现方法 `dump` 来导出数据集。该访问内部应该调用 `BaseBuilder.build()` 来构建数据集，并实现将数据集写入到磁盘的逻辑。

* 数据集参数文件

  自定义子类的 `dump` 方法还应该导出数据集相关参数到文件，如 `datasets.json`。

- 保证分布一致性

  训练集上学习到的模型能够具有较好泛化能力的一个重要前提条件是：训练集的数据分布跟真实的数据分布尽可能的一致。因此划分数据源之前，务必要对其进行随机打乱（shuffle），以保证各个数据集分布是一致的。打乱的逻辑已经由 `BaseBuilder.build` 实现，子类只需要调用它即可。

- 懒惰载入数据

  在某些项目中数据源可能特别大，无法一次载入内存中，但同时我们又想要将其随机打乱，该怎么办？建议在 `load` 时只是加载数据的索引以及存储位置，然后在 `dump` 时再读取数据并划分到相应的数据集中。

- 构建小数据集

  如果想要在正式训练模型之前，先使用小批量的数据来验证模型的可行性，可以通过为 `BaseBuilder` 的参数 `data_factor` 来指定希望构建数据源的比例。

## lib.dataset.loader

载入数据集并变换为模型可用的格式。

### lib.dataset.DatasetHandler

从磁盘载入数据集，需要实现一个 Handler，它每次读入一个样本。为了支持大数据集的读取，应该将数据集随机打乱的逻辑放在此处。

应该重写的方法：

* `__init__()` 初始化 Handler
* `__enter__()` 进入 Handler
* `__exit__()` 清理 Handler
* `read()` 返回一个样本或者 `None`。`None` 用以表示数据集尾部 ；样本的格式根据应用来自定义。

### lib.dataset.DatasetIterator

一般情况下，不需要重写此类。支持以样本（sample）为单位对数据集进行迭代。虽然该类提供了 `shuffle` 以支持乱序访问样本，但不适用于大数据集，推荐通过 Handler 来实现乱序访问的逻辑。

支持的属性和方法：

* `__init__(name, size, handler, shuffle=False, transform=None)` 初始化

* `name` 数据集名字
* `size` 数据集大小

### lib.dataset.BatchIterator

一般情况下，不需要重写此类。支持以批（batch）为单位对数据集进行迭代。这种数据迭代方式也是最为常见的情形。

支持的属性和方法：

* `__init__(dataset, batch_size=1, transform=None)` 初始化
* `batch_size` 批大小
* `dataset_size` 数据集大小
* `dataset_name` 数据集名字

### lib.dataset.BaseLoader

项目需要根据自己的需求，自定义 Loader 类。需要为该类提供 Handler 和 Transform，然后可以得到 DatasetIterator 或 BatchIterator。

#### Description

支持的属性和方法：

* `__init__(sample_transform=None, batch_transform=None)` 初始化
* `load(handler, name, size, batch_size=None, shuffle=False, sample_transform=None, batch_transform=None)` 加载数据集
* `create_loader(*args, **kwargs)` 生成数据集加载器

#### Usage

需要自定义样本变换类和批变换类。

重载 `DatasetHandler` 实现自定义的数据集读取类。

继承 `BaseLoader` 实现自定义数据集加载类，主要需要实现 `load()` 方法。

## lib.dataset.transform

用于数据集的变换。

项目需要根据自己的需求来自定义变换类。变换类可以是任何函数或者实现了 `__call__` 方法的自定义类。

### lib.dataset.transform.ComposeTransform

用于将多个变换组合为一个。在想要对样本或者批进行多个变换时，可以利用该类将它们组织为一个变换。

# lib.training

## lib.training.get_parser

解析训练参数的命令行解析器。

## lib.training.pipeline

### lib.training.pipeline.Pipeline

实现训练过程的抽象逻辑。

#### Description

支持的属性和方法：

*  `action_before_run(context={})`
* `action_before_epoch(context={})`
* `action_before_train(context={})`
* `action_before_evaluate(context={})`
* `action_after_evaluate(context={})`
* `action_after_train(context={})`
* `action_after_epoch(context={})`
* `action_after_run(context={})`
* `train(trainset, num_batches, epoch)` 
* `evaluate(dateset)`
* `run(context={})` 开始运行 Training pipeline

#### Usage

该 Pipeline 提供了一个模型训练过程的抽象逻辑，用伪代码描述如下：

```
for epoch in range(num_epochs):
	# train phase
	model.set_train_mode()
	for batch in train_set:
		inputs, targets = batch
		# forward
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		# backward
		loss.backward()
		# update parameters
		update_model(model.parameters)
	
	# evaluate phase
	model.set_evaluate_mode()
	evaluate_metrics = []
	for batch in val_set:
		inputs, targets = batch
		stat = {}
		with no_grad():
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			stat['loss'] = loss.item()
			for name, metric in metrics:
				stat[name] = metric(outputs, targets)
		evaluate_metrics.append(stat)	
```

一般来说项目中是不需要重写该训练抽象逻辑的。如果需要自定义训练过程的行为，可以考虑看能否通过重写 `action_*` 之类的方法来实现。

# lib.evaluation

## lib.evaluation.get_parser

解析评估的命令行解析器。

## lib.evaluation.pipeline

### lib.evaluation.pipeline.Pipeline

实现评估过程的抽象逻辑。

# lib.experiment

## lib.experiment.dataset_cfg

数据集基本信息配置。

## lib.experiment.experiment_cfg

实验基本信息配置。

## lib.experiment.summay

汇总实验结果。

# lib.hyperparam_optim

## lib.hyperparam_optim.gird_search

超参的 Grid search 优化策略。

# lib.utils



