# COMP_7404_2025_spring_LoRA
此仓库储存的是2025春季COMP7404的项目，主题为验证LoRA微调方式的有效性以及论文中提出的部分优势       
核心论文为：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)        
测试使用的底层模型为：[RoBERTa base model](https://huggingface.co/FacebookAI/roberta-base)， 模型参数为：125M params       
项目成员：刘知闲(组长), 田潇文，张维轩， 张哲      
#### 2025.3.1  项目分工
1. 理解并加载MNLI matched，STSB两个数据集，若发现更优的测试集可以改成对应数据集，并同时负责DataLoader的编写及loss function的编写，最后辅助其他部分的同学进行训练时数据的使用. (田潇文)
2. 验证LoRA有效性，验证不同rank的影响，用Roberta模型在1中两个数据集上测试微调时间，准确率/分数(取最优的rank值)和LoRA在小批量预测中的计算速度。(张哲)
3. 验证LoRA微调运用在attention中q，v，k，o四个可优化矩阵上的不同影响，测试超参数维持和论文中一致。(刘知闲) 
4. 以2中最优rank下的LoRA同参数的adapter方式以及全参数的方式，用Roberta模型进行1中两个数据集上测试微调时间，准确率/分数以及在小批量预测中的计算速度。(张维轩)
5. 项目框架设计，考虑用parser进行统筹，使参数输入以及每个部分编写更有条理。 (刘知闲)

#### 项目进度
* 2025.3.2 项目框架初步构建, 可以开始数据加载以及本体模型的编写。

* 2025.3.2 因为考虑到项目整体的修改集中在main.py文件中，此项目采用branch+pull request 方式进行更新

* 2025.3.7 验证rank功能上线，等待实验取最优rank，再进行下一步小批量预测。

* 2025.3.11 LoRA测试部分可以成功运行，但Roberta-base模型似乎无法有效完成stsb任务，微调后结果不尽人意，需要继续试验。

* 2025.3.17 更新了ag_news数据集，修改了数据集的加载方式（不删除label列），解决了2025.3.11的问题。

