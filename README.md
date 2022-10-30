# CCKS 2022 NL2SQL

### 依赖环境

```
pip install -r requirements.txt
```
本地使用的pytorch版本是torch-1.11.0+cu113。**torch-1.12经验证可能无法在3090上使用**。

### 配置文件

本项目使用[Hydra](hydra.cc)进行项目管理。`conf`目录下包含了所有实验的配置文件，细节如下：
> - `conf/dpr/base_v2_0` DPR
> - `mt5/xl/base_cls_v1_1_adafactor` End2End generation (baseline)
> - `mt5/xl/cls_bm25_table_cn_v1_2` mt5 + table names BM25 这里名字写错了 其实拼接的是英文表名(en)而非中文(cn)
> - `mt5/xl/cls_bm25_table_cn_col_en_v1_0` mt5 + table names BM25 + column names
> - `mt5/xl/cls_bm25_table_dpr_en_col_en_v2_1` mt5 + table names DPR + column names
> - `mt5/xl/cls_bm25_table_dpr_en_v2_0` mt5 + table names DPR
> - `mt5/xl/cls_bm25_table_dpr_en_v2_0_add_cn` mt5 + table names DPR (en + cn)
> - `mt5/xl/cls_bm25_table_dpr_en_col_aug_vxxx` mt5 + table names DPR + column names + data augmentation (entity/header replacing)
> - `mt5/xl/cls_bm25_table_dpr_en_col_en_gd_v2_0_fix_slurm` mt5 + table names DPR + column names + structure grounding
> - `mt5/xl/cls_table_dpr_en_col_en_gd_example_v5_0_slurm` mt5 + table names DPR + column names + retrieved examples (using BM25)
> - `mt5/xl/cls_table_dpr_en_col_en_gd_exp_xxx` 同上，在全量数据集上训练，基于同样在全量数据集上使用非retrieval的预训练后的模型
> - `mt5/xl/cls_table_dpr_en_col_en_gd_v2_0_full2_type` 增加了关于实体和实体属性的type embedding
> - `du_sql/mt5-xl/table_cn_col_cn_gd_v1_0_slurm` mt5 + table names + columns names + structure grounding ON DuSQL Dataset

启动命令（以baseline为例）：
```
srun -N 1 -n 1 -c 4 --gpus-per-task 4 --mem 40G python trainer_slurm_fsdp.py -cp conf/mt5/xl -cn base_cls_v1_1_adafactor
```
`trainer_slurm_fsdp.py`也支持分布式数据并行训练，需要设置`-n 4`，同时开启cpu_offload才可以训练mt5-xl。DPR训练因模型较小，不需要模型并行，可以使用数据并行。

### 数据文件说明

数据文件保存在`ccks2022/ccks_nl2sql`目录下
>- `train.json` 全量训练集，含标注。
>- `dev.json` 验证集，用于日常提交，无标注。
>- `dev_gold.json` 测试阶段公布的验证集，数据与`dev.json`一致，包含标注。
>- `sub_train.json` 日常提交阶段因无验证集，从`train.json`选出了一部分数据当做验证集后剩下的训练集部分。
>- `sub_dev.json` 同上，从`train.json`中选的部分数据，当做验证集。
>- `sub_dev_0626.json` 纠正了`sub_dev.json`中部分标注错误（大概7条）
>- `train_4466.json` 从`dev_gold.json`中随机挑选了500条数据补充后的训练集。
>- `dev_gold.json` 从`dev_gold.json`中随机挑选了部分数据扩充训练集后剩余的验证集。
>- `test.json` 测试集，无标注。
>- `data_combine.json` `train.json` + `dev_gold.json`，用于测试阶段作为基于检索的模型的数据库。
>- `fin_kb.json` 包含了数据库中的各种实体、实体的属性名，以及实体属性的枚举值。也可自行从数据库中进行过滤。
>- `db_info.json` 数据库信息，包含各个数据库的表、列和链接键等信息。

### 数据处理

本项目的核心在于数据处理部分
> - `data/bm25.py` BM25 model
> - `data/data_utils.py` 数据处理的一些工具函数
> - `data/du_sql.py` 解析DuSQL标注结构体并构造输入输出
> - `data/entity_label_utils.py` 利用知识图谱文件，对输入中的实体属性枚举值和实体属性名进行标注，增加额外的type embedding用于训练
> - `data/multi_task_grounding.py` structure grounded training
> - `data/retrieval.py` 构造训练检索模型需要的数据
> - `data/seq2seq.py` 构造纯生成时所需要的数据


### 关于DPR

DPR用于训练检索与自然语言查询相关的表名，也可用于进一步查询相关的列名（未实现）。

#### Training

指定config`-cp conf/dpr -cn base_v2_0`即可加载训练，需要注意设置`do_train: True`以及正确的`test_file`。

由于CCKS数据库中包含的表名较少，因此目前的训练方法是对query编码得到向量表示后，与所有候选表名的向量表示计算相似度，使用交叉熵优化并更新参数。
前向计算时同时更新所有表名的表示。

对于表名较多的情况，可选的使用in-batch negative sampling，即在batch内部进行负采样，注意需要对mini-batch内部其他正确的表名进行mask。
相关的数据读取逻辑可以参考`data.retrieval.table_name_ranking`。

#### Inference

修改`do_train: False`，同时指定`test_file`即可。需要注意目前没有实现指定预测结果文件的保存文件名，因此需要手动在每次预测完成后将`输出目录/test/eval_predictions.json`修改为可区别的文件名。


### 评价指标

CCKS NL2SQL 使用`exact match`作为唯一指标，另在赛事公开了具体的指标计算方法后（主要指忽略结构体中某些关键字下内容的顺序），本代码也进行了更新，如需使用，需要指定`post_process`部分指定如下内容：

```
post_process:
  _target_: general_util.metrics.SQLResultsWClsHelper
  official_evaluate: True
```
此时返回的指标中`cls_em`指标即为官方使用的`em`评测指标。

#### 其他指标

在训练过程中也使用了一些其他的指标方便监控训练状态，包括：
> - `xxx_acc` 某项分类任务的准确率
> - `acc` teacher forcing状态下decoding的逐token准确率（仅在训练过程中的验证时可用）
> - `BLEU` 对生成序列使用了BLEU进行评价，需要注意的时分词使用的是nltk，对中文会有问题（可以换成jieba）。
> - `em` 不包含数据库分类的`exact match`指标。需要注意的是计算过程没有考虑某些关键词内容的顺序可以忽略不计，因此指标较实际情况偏低，仅供参考。


#### 实现细节

指标的计算有两种实现方式，一种是耦合在模型的代码中，对于一些比较简单的指标（比如准确率）可以使用，需要参考的代码部分：

> - `general_util.mixin.LogMixin` 模型层面的接口，用于初始化metric
> - `general_util.average_meter.LogMetric` metric类，统一封装了metric（目前只实现了平均）

另一种是实现`post_processor`，用于后处理，`post_processor`需要实现`__call__(meta_data, batch_outputs)`和`get_results()`方法，
前者用于对单个mini-batch的模型输出进行处理、记录，或结合原始数据（meta_data，如字符串等无法张量化的标签等）进行后处理，后者返回整个验证集上的指标和预测结果，
对于一些不依赖全数据集的指标，可以在`__call__`方法中计算mini-batch的指标，并在`get_results`方法中平均，对于依赖全数据集的指标计算，建议在`__call__`方法中只对预测结果和标签进行保存。

指标计算支持数据平行时的多卡分布式evaluation，需要实现`post_processor`类并实现`general_util.mixin.DistGatherMixin`类，具体可参考`general_util.metrics.DuSQLResultsHelper`。


### 关于Hydra的特性

Hydra通过`hydra.utils.call`方法，可以从配置文件初始化类或调用特定函数，并将返回值作为参数或输入。

以`conf/dpr/base_v2_0.yaml`配置文件为例，在模型初始化时`model`域下`_target_`指定了要调用的函数或类，在`trainer_slurm_fsdp_v1.py`的第475行将其返回值（对象）赋值给`model`引用。

可以避免代码不同模块之间耦合程度过高并降低增加新模块的改动难度。在尝试理解代码时可以根据配置文件找到调用的具体方法。

[//]: # (### TODO List)

[//]: # (- Multiple training set.)

[//]: # (- Resume training from checkpoint.)

[//]: # (  - Different setup for different engines, e.g., fairscale and deepspeed, or vanilla pytorch.)
