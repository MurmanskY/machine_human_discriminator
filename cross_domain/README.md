# Cross-domain

使用CH3-zh数据集，该数据集由ChatGPT-3.5生成，包含7个domain。

每次剔除一个domain，在剩下六个domain上进行训练，在剔除的那一个domain上进行测试。

指标使用F1-score