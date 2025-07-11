# 手写数字识别

## 项目简介

基于简单的神经网络实现手写数字识别
layer 1:25 unit, using relu as activation function
layer 2:15 unit, using relu as activation function
layer 3:10 unit, output layer,softmax activation function

在mnist测试集下，准确率大约96%

## 文件介绍

`model`:存放模型的文件夹，模型为`model.keras`
`pic`:放数字识别的文件夹，注意，图片要png格式，28*28大小
`train`:训练模型用
`application`:主程序，识别数字，将结果输出到控制台

## 使用

### 训练

运行`train`文件夹下的`training.py`,完成后自动保存模型和报告，在控制台也能看到报告

### 运行

运行`application.py`,结果会输出到控制台

## 不足之处
- 对于数据集之外的数字识别可能表现较差
- 对于不是数字的图片也可能误判为数字

