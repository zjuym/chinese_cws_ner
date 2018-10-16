### 引用

IDCNN+CRF参考论文：[https://arxiv.org/abs/1702.0209](https://arxiv.org/abs/1702.02098)

### 语料

采用人民日报语料(下载地址: <https://pan.baidu.com/s/1crEN5G> ，密码：kbp2).

将语料中以/nr和/nt结尾的词随机替换为现有库中的名字和组织结构名.

### 需要的环境

tensorflow>=1.3.1

keras==2.1.3

keras_contrib==0.0.2

### 训练


##### 训练分词

python train_model.py --model=cws --which_model=idcnn --data_path=<绝对路径到分词训练数据>../data/small_cws_train_data.txt --epoch=5 --ner_model=../model_dir/cws_model.h5

##### 训练命名实体识别

python train_model.py --model=ner --which_model=idcnn --data_path=<绝对路径到命名实体识别训练数据>../data/small_ner_train_data.txt --epoch=5 --ner_model=../model_dir/ner_model.h5

### 使用

python demo.py

