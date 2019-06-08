# MNIST ink

## 主要功能
    
    墨迹识别

## 实现原理

    使用tensorflow构建卷积神经网络（CNN）从而实现对0-9的数字的图片分类识别，导出pb model之后，使用C#设计用鼠标代替手写的界面，导入pb model进行识别

1. 任务分配

    编程设计人员3人
    - 董明超 training层、evaluation层，流程架构整合
    - 周文华 inference层、loss层
    - 吴延彬 数据输入层、数据集

2. 文件结构

    完整的mnist分为四个部分，分别是inference()、loss()、training()、evaluation()
    根据此分为四个文件
    - inference.py
    - loss.py
    - training.py
    - evaluation.py
    功能性文件
    - input_data.py

3. 参考资料
   
    - inference层独立的写法： https://zhuanlan.zhihu.com/p/27552202
    - 分层建议： https://blog.csdn.net/NNNNNNNNNNNNY/article/details/58104287
    - 在GitHub的 samples-for-ai\examples\tensorflow\MNIST 中的内容
    - pb文件的保存和读取：https://zhuanlan.zhihu.com/p/32887066