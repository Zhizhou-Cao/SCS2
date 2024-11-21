# SCS2
Statistical Case Study Assignment 2

Workflow:
1. Create the models using KNN, DA and Random forest. 
2. Find the accuracy of each model.
3. Determine if the length of paper affects accuracy of models.
4. Check if different topic affects accuracy of models, by creating a model that does not include 'Art', for example. 

Q1. Machine or Human ? (1000 words models) [N-fold cross validation]
合成两个新的文件夹，人和机器

Q2. 测试不同模型在不同数据集的表现
三个训练模型【500/750/1000】
三个测试集【500/750/1000】
出一个3x3的矩阵table

Q3. two models(train: 1.whole train set/ 2.except "Architecture").  test set always Architecture
  1. train: 纯"Architecture"; test: "Architecture"
  2. train: 全部剩下 ; test: "Architecture"
导致error的原因：features有一整个column的数字一样，导致normalisation=NA，所以报错
  4. train: 全部剩下 + "Architecture"; test: "Architecture"
  
  
Q4. [undecided] small function words


report:
methodology part: 可以复制上一次作业的内容，不查重（但要闲的没事干可以rewrite)
最新教的SVM，没说要写到report里面，但用了可能会分多一些？
