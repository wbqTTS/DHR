import csv
import time
import threading
import pandas as pd
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
import warnings


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
def dispatch_predictions(models, num=3):
    """
    随机选择三个预测结果
    :param predictions: 预测结果的字典
    :param num: 要选择的数量
    :return: 选择的预测结果 (01 列表)
    """
    selected_keys = random.sample(list(models.keys()), num)  # 随机选择键
    selected = {key: models[key] for key in selected_keys}  # 选出对应的键值对

    # 打印选择的 key 和对应的值
    print("Selected models:")
    for key, value in selected.items():
        print(f"model: {key}")

    return list(selected.values())

def majority_decision(results):
    # 多模裁决机制：基于多数投票来判断是否异常流量
    print("多模裁决...")
    decision = sum(results) >= len(results) / 2  # 如果超过一半为异常，则认为是异常
    print(f"裁决结果：{'异常流量' if decision == 1 else '正常流量'}")
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
            print(f"当前流量Features: {features}")

            # 使用 SVM 模型预测
            svm_prediction = svm_model.predict([features])[0]
            # 使用 随机森林 模型预测
            rf_prediction = rf_model.predict([features])[0]
            # 使用 knn 模型预测
            knn_prediction = knn_model.predict([features])[0]
            # 使用 普通 模型预测
            normal_prediction = int(icmp_count_perSec > threshold)

            print(f"SVM 模型预测: {svm_prediction}, 随机森林模型预测: {rf_prediction}, "
                  f"knn模型预测：{knn_prediction}, 普通预测："
                  f"{normal_prediction}")

            models = {
                "svm": svm_prediction,
                "rf": rf_prediction,
                "knn": knn_prediction,
                "normal": normal_prediction
            }

            # 调度算法
            selected_predictions = dispatch_predictions(models, num=3)

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

# 执行 ping 命令的线程函数
def ping_host(host, cmd):
    try:
        host.cmd(cmd)
    except Exception as e:
        print(f"Ping 命令执行失败: {e}")

# 生成数据集
def generate_dataset(h1, h2, s1, duration=20, interval=2):
    print("开始模型训练...")

    # 创建一个 CSV 文件来保存数据集
    with open('traffic_dataset.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['icmp_count', 'byte_count', 'rate', 'label'])

        # 捕获正常流量数据集
        print("收集正常流量数据（ping流量）...")
        interface = s1.intf('s1-eth1').name  # 获取网络接口名称

        # 定义 ping 命令
        normal_ping_cmd = f'timeout {duration + 2}s ping {h2.IP()}'

        # 启动线程执行 ping 命令
        ping_thread = threading.Thread(target=ping_host, args=(h1, normal_ping_cmd))
        ping_thread.start()

        # 捕获流量
        start_time = time.time()
        while time.time() - start_time < duration:
            packets = sniff(filter="icmp", iface=interface, timeout=2)
            icmp_count = len([pkt for pkt in packets if pkt.haslayer('ICMP')])
            byte_count = sum(len(pkt) for pkt in packets)
            rate = icmp_count / interval
            writer.writerow([icmp_count, byte_count, rate, 0])
            print(f"正常流量：icmp_count={icmp_count}, byte_count={byte_count}, rate={rate}")

        # 确保 ping 线程完成
        ping_thread.join()

        # 捕获流量（无流量时，也属于正常流量）
        print("收集正常流量数据（无流量）...")
        start_time = time.time()
        while time.time() - start_time < duration:
            packets = sniff(filter="icmp", iface=interface, timeout=2)
            icmp_count = len([pkt for pkt in packets if pkt.haslayer('ICMP')])
            byte_count = sum(len(pkt) for pkt in packets)
            rate = icmp_count / interval
            writer.writerow([icmp_count, byte_count, rate, 0])
            print(f"正常流量：icmp_count={icmp_count}, byte_count={byte_count}, rate={rate}")

        # 捕获异常流量数据集
        print("收集异常流量数据...")
        abnormal_ping_cmd = f'timeout {duration*2 + 4}s ping -f {h2.IP()}'

        # 启动线程执行 ping -f 命令
        ping_thread = threading.Thread(target=ping_host, args=(h1, abnormal_ping_cmd))
        ping_thread.start()

        # 捕获流量
        start_time = time.time()
        while time.time() - start_time < duration*2:
            packets = sniff(filter="icmp", iface=interface, timeout=2)
            icmp_count = len([pkt for pkt in packets if pkt.haslayer('ICMP')])
            byte_count = sum(len(pkt) for pkt in packets)
            rate = icmp_count / interval
            writer.writerow([icmp_count, byte_count, rate, 1])
            print(f"异常流量：icmp_count={icmp_count}, byte_count={byte_count}, rate={rate}")

        # 确保 ping 线程完成
        ping_thread.join()

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
    X = data[['icmp_count', 'byte_count', 'rate']]  # 特征列
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

    print("发起 h1 ping -f h2(模拟ddos攻击)")
    h1.cmd('ping -f %s' % h2.IP())

    # 等待监控线程结束
    monitor_thread.join()

    print("关闭网络拓扑...")
    net.stop()
    time.sleep(5)  # 休眠 5 秒
    print("实验结束！")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    run()
