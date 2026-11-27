"""MNIST 手写数字识别项目的源码包。

模块划分（按 environment.md 的工程结构）：
    - models.py     CNN 与全连接对照网络
    - data.py       MNIST 读取、预处理和 DataLoader
    - train.py      设备选择、训练循环、资源计量和权重保存
    - evaluate.py   测试、可视化和单图预测
    - report.py     多种子汇总、差值表和 CSV/Markdown 输出
    - gui.py        Tkinter 单图预测与实验对比界面
    - utils.py      运行目录、JSON 和环境信息
"""
