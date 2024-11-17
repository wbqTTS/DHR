import csv
import time
import threading
from turtle import pd

import joblib
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.link import TCLink
from scapy.all import sniff
import requests
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random


class MyTopo(Topo):
    # 构建拓扑结构
    def build(self):
        s1 = self.addSwitch('s1')
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)

# 随机调度三个预测结果
def dispatch_predictions(predictions, num=3):
    """
    随机选择三个预测结果
    :param predictions: 预测结果的字典
    :param num: 要选择的数量
    :return: 选择的预测结果 (01 列表)
    """
    selected = random.sample(list(predictions.values()), num)
    return selected

def majority_decision(results):
    # 多模裁决机制：基于多数投票来判断是否异常流量
    decision = sum(results) >= len(results) / 2  # 如果超过一半为异常，则认为是异常
    return decision

def monitor_icmp_realtime(interface, controller_ip, h1_ip, h2_ip, threshold,
                          svm_model_file, rf_model_file, knn_model):
    # 加载 SVM 和随机森林模型
    svm_model = joblib.load(svm_model_file)
    rf_model = joblib.load(rf_model_file)
    print(f"启动实时监控接口 {interface} 的 ICMP 包流量...")
    while True:
        try:
            # 捕获数据包，统计 ICMP 包数量
            packets = sniff(filter="icmp", iface=interface, timeout=2)
            icmp_count = len([pkt for pkt in packets if pkt.haslayer('ICMP')])
            byte_count = sum(len(pkt) for pkt in packets)
            # 单位时间为 2 秒
            icmp_count_perSec = icmp_count / 2
            print(f"接口 {interface} 每秒接收到的 ICMP 包数量: {icmp_count_perSec}")

            # 提取特征用于机器学习模型预测
            features = [icmp_count, byte_count, icmp_count_perSec]

            # 使用 SVM 和随机森林模型预测
            svm_prediction = svm_model.predict([features])[0]
            rf_prediction = rf_model.predict([features])[0]
            knn_prediction = knn_model.predict([features])[0]
            normal_prediction = icmp_count_perSec > threshold
            print(f"SVM 模型预测: {svm_prediction}, 随机森林模型预测: {rf_prediction}, "
                  f"knn模型预测：{knn_prediction}, 普通预测："
                  f"{normal_prediction}")

            predictions = {
                "svm": svm_prediction,
                "rf": rf_prediction,
                "knn": knn_prediction,
                "normal": normal_prediction
            }

            # 调度算法
            selected_predictions = dispatch_predictions(predictions, num=3)

            # 多模测试
            if majority_decision(selected_predictions):
                print("检测到异常流量，开始阻断...")
                block_ping(controller_ip, h1_ip, h2_ip)
                break  # 一旦阻断，停止监控
        except Exception as e:
            print(f"监控时发生错误: {e}")
            break


def block_ping(controller_ip, h1_ip, h2_ip):
    flow_rule = {
        "dpid": 1,
        "table_id": 0,
        "idle_timeout": 0,
        "hard_timeout": 0,
        "priority": 65535,
        "flags": 1,
        "match": {
            "in_port": 1,
            "dl_type": 2048,
            "nw_proto": 1
        },
        "actions": []
    }
    response = requests.post(f"http://{controller_ip}:8080/stats/flowentry/add", json=flow_rule)
    if response.status_code == 200:
        print("成功下发流表规则，阻止 h1 ping 通 h2")
    else:
        print("流表规则下发失败:", response.text)


# 用于持续生成数据集的函数
def generate_dataset(h1, h2, s1, duration=20, interval=2):
    print("开始模型训练...")

    # 创建一个 CSV 文件来保存数据集
    with open('traffic_dataset.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入 CSV 文件的表头
        writer.writerow(['icmp_count', 'byte_count', 'rate', 'label'])

        # 1. 正常流量数据集： h1 ping h2
        print("收集正常流量数据...")

        # 启动一个新的线程，持续收集 ICMP 包数据
        start_time = time.time()

        # 启动 ping 命令
        h1.cmd('timeout 22s ping %s' % h2.IP())

        while time.time() - start_time < duration:
            # 捕获数据包
            interface = s1.intf('s1-eth1').name
            packets = sniff(filter="icmp", iface=interface, timeout=interval)

            # 提取特征
            icmp_count = len([pkt for pkt in packets if pkt.haslayer('ICMP')])
            byte_count = sum(len(pkt) for pkt in packets if pkt.haslayer('ICMP'))
            rate = icmp_count / interval

            # 将数据写入 CSV 文件（正常流量标签为 0）
            writer.writerow([icmp_count, byte_count, rate, 0])
            print(f"正常流量：icmp_count={icmp_count}, byte_count={byte_count}, rate={rate}")


        # 2. 异常流量数据集： h1 ping -f h2
        print("收集异常流量数据...")

        # 启动 ping -f 命令
        start_time = time.time()
        # 发起频繁的 ping
        h1.cmd('timeout 22s ping -f %s' % h2.IP())


        while time.time() - start_time < duration:
            # 捕获数据包
            packets = sniff(filter="icmp", iface=interface, timeout=interval)

            # 提取特征
            icmp_count = len([pkt for pkt in packets if pkt.haslayer('ICMP')])
            byte_count = sum(len(pkt) for pkt in packets if pkt.haslayer('ICMP'))
            rate = icmp_count / interval

            # 将数据写入 CSV 文件（异常流量标签为 1）
            writer.writerow([icmp_count, byte_count, rate, 1])
            print(f"异常流量：icmp_count={icmp_count}, byte_count={byte_count}, rate={rate}")

    print("数据集生成完毕，已保存为 'traffic_dataset.csv'")



#-----------------------k邻近算法--------------------------------
# 定义训练 KNN 模型的函数
def train_knn_model(csv_file, k=3):
    """
    使用 KNN 算法训练模型
    :param csv_file: 数据集文件路径 (CSV 格式)
    :param k: KNN 中的邻居数量
    :return: 训练好的模型
    """
    print("加载数据集...")
    data = pd.read_csv(csv_file)

    # 数据预处理
    X = data[['packet_count', 'byte_count', 'rate']]  # 特征列
    y = data['label']  # 标签列

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 KNN 模型
    print("训练 KNN 模型...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = knn.predict(X_test)
    print("KNN 模型评估:")
    print("准确率:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return knn




#----------------------random_forest------------------------
def train_random_forest_model(csv_file):
    # 加载数据集
    df = pd.read_csv(csv_file)

    # 提取特征和标签
    X = df[['icmp_count', 'byte_count', 'rate']]  # 选择特征列
    y = df['label']  # 选择标签列

    # 划分数据集为训练集和测试集（训练集占比 80%，测试集占比 20%）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"随机森林模型的准确率: {accuracy:.2f}")

    # 保存模型到文件
    joblib.dump(rf_model, 'random_forest_model.pkl')
    print("模型训练完成并保存为 'random_forest_model.pkl'")


def predict_with_random_forest_model(csv_file, model_file='random_forest_model.pkl'):
    # 加载数据集
    df = pd.read_csv(csv_file)

    # 提取特征
    X = df[['icmp_count', 'byte_count', 'rate']]

    # 加载训练好的模型
    rf_model = joblib.load(model_file)

    # 进行预测
    y_pred = rf_model.predict(X)

    # 输出预测结果
    print(f"预测标签：{y_pred}")
    return y_pred






# ---------------------svm------------------------
# svm模型预测是否异常流量
def train_svm_model(csv_file):
    """
    SVM 模型训练函数
    :param csv_file: 数据集文件名
    """
    # 加载数据集
    df = pd.read_csv(csv_file)

    # 提取特征和标签
    X = df[['icmp_count', 'byte_count', 'rate']]  # 特征列
    y = df['label']  # 标签列

    # 划分数据集为训练集和测试集（80%训练集，20%测试集）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练 SVM 模型
    svm_model = svm.SVC(kernel='linear', C=1.0)  # 线性核函数，默认正则化参数 C=1.0
    svm_model.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM 模型的准确率: {accuracy:.2f}")

    # 保存模型到文件
    joblib.dump(svm_model, 'svm_model.pkl')
    print("模型训练完成并保存为 'svm_model.pkl'")


def predict_with_svm_model(csv_file, model_file='svm_model.pkl'):
    """
    使用训练好的 SVM 模型进行预测
    :param csv_file: 数据集文件名
    :param model_file: 训练好的模型文件名
    :return: 预测结果
    """
    # 加载数据集
    df = pd.read_csv(csv_file)

    # 提取特征
    X = df[['icmp_count', 'byte_count', 'rate']]

    # 加载训练好的 SVM 模型
    svm_model = joblib.load(model_file)

    # 进行预测
    y_pred = svm_model.predict(X)

    # 输出预测结果
    print(f"预测标签：{y_pred}")
    return y_pred




def run():
    topo = MyTopo()
    net = Mininet(topo=topo, controller=lambda name: RemoteController(name, ip='127.0.0.1'), link=TCLink)
    net.start()

    # 禁用所有主机的 IPv6
    for host in net.hosts:
        host.cmd('sysctl -w net.ipv6.conf.all.disable_ipv6=1')
        host.cmd('sysctl -w net.ipv6.conf.default.disable_ipv6=1')
    print("已禁用所有主机的 IPv6")

    # 获取主机和交换机
    h1, h2 = net.get('h1', 'h2')
    s1 = net.get('s1')

    # 训练机器学习模型
    generate_dataset(h1, h2, s1)
    csv_file = 'traffic_dataset.csv'
    train_svm_model(csv_file)
    train_random_forest_model(csv_file)
    svm_model_file = 'svm_model.pkl'
    rf_model_file = 'random_forest_model.pkl'
    knn_model = train_knn_model(csv_file)

    # 启动监控线程
    monitor_thread = threading.Thread(
        target=monitor_icmp_realtime,
        args=(s1.intf('s1-eth1').name, '127.0.0.1', h1.IP(), h2.IP(), 3, svm_model_file, rf_model_file, knn_model)
    )
    monitor_thread.start()


    # 启动 h1 ping h2 10 秒后自动停止
    print("发起 h1 ping h2(模拟正常流量), 10 秒后自动停止...")
    h1.cmd('timeout 10s ping %s' % h2.IP())

    # 等待 13 秒后启动高频 ping
    time.sleep(13)
    print("发起 h1 ping -f h2(模拟ddos攻击)")
    h1.cmd('ping -f -c 100 %s' % h2.IP())

    # 等待监控线程结束
    monitor_thread.join()

    net.stop()


if __name__ == '__main__':
    run()
