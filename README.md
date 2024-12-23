### 核心功能

1. **网络拓扑构建**：
   - 使用 `MyTopo` 类定义了一个简单的拓扑，包含一个交换机 `s1` 和三个主机 `h1`, `h2`, `h3`。
   - 利用 Mininet 创建网络并禁用主机的 IPv6。
2. **数据生成**：
   - 使用 `generate_dataset` 方法模拟网络流量，分为正常流量 (`ping`) 和异常流量 (`ping -f`) 两种情况。
   - 将捕获到的 ICMP 包数量、字节数和速率作为特征，生成带有标签的流量数据集并保存为 CSV 文件。
3. **机器学习模型训练**：
   - 提供了 SVM、随机森林和 KNN 模型的训练功能。
   - 加载数据集、提取特征、训练模型并评估其准确性，同时将模型保存为 `.pkl` 文件以供后续使用。
4. **实时监控与预测**：
   - 在实时监控线程中捕获接口上的 ICMP 包流量。
   - 基于统计特征，使用预训练的 SVM、随机森林和 KNN 模型进行预测。
   - 结合调度算法（随机选择 3 个模型预测结果）和多模裁决（多数投票机制）来做最终决策。
5. **阻断流量**：
   - 当检测到异常流量时，通过 REST API 向 Ryu 控制器下发流表规则，阻止特定的 ICMP 流量。

### 需要注意的事项

1. **依赖库**：
   - 代码中用到了以下 Python 库：
     - Mininet (`mininet`)
     - Scapy (`scapy.all.sniff`)
     - Scikit-learn (`sklearn`)
     - Joblib (`joblib`)
     - Pandas (`pandas`)
     - ......
   - 确保你已经安装了所有依赖库。
2. **模型性能**：
   - 数据集生成的样本有限，可能不足以支持训练高性能模型。。
3. **流量特征选择**：
   - 当前特征为 ICMP 包数量、字节数和速率，可能不够丰富。
