import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# 1.加载数据并按比例划分
def load_and_split():
    """加载iris数据集，按照5:2:3划分训练，验证，测试集"""
    iris = load_iris()
    X,y = iris.data,iris.target
    class_names = iris.target_names

    # 划分训练集和临时集
    X_train,X_temp,y_train,y_temp = train_test_split(
        X,y,test_size=0.5,random_state=42
    )

    # 划分验证集和测试集
    X_dev,X_test,y_dev,y_test = train_test_split(
        X_temp,y_temp,
        test_size=0.6,
        random_state=42
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_dev.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    return X_train,X_dev,X_test,y_train,y_dev,y_test,class_names

# 2.训练并评估不同参数组合
def evaluate_knn_parameters(X_train,y_train,X_dev,y_dev,k_values,metrics):
    """评估不同参数组合的KNN模型"""
    results = []

    for metric in metrics:
        for k in k_values:
            # 训练模型
            knn = KNeighborsClassifier(n_neighbors=k,metric=metric)
            knn.fit(X_train,y_train)

            # 在cross validation set上评估
            dev_pred = knn.predict(X_dev)
            dev_acc = accuracy_score(y_dev,dev_pred)

            results.append({
                'metric': metric,
                'k': k,
                'dev_accuracy': dev_acc,
                'model': knn
            })
            print(f"距离度量: {metric}, K={k}, 验证集准确率: {dev_acc:.4f}")

    # 找到最好的模型
    best_idx = np.argmax([r['dev_accuracy'] for r in results])
    best_model = results[best_idx]
    print(f"\n最佳参数: 距离度量={best_model['metric']}, K={best_model['k']}, 验证集准确率={best_model['dev_accuracy']:.4f}")

    return results,best_model

# 3.在测试集上评估最佳模型
def evaluate_best_model(best_model,X_test,y_test):
    """用最佳模型评估"""
    test_pred = best_model['model'].predict(X_test)
    test_acc = accuracy_score(y_test,test_pred)
    print(f"测试集准确率: {test_acc:.4f}")
    result = classification_report(y_test, test_pred, target_names=class_names)
    print(result)
    with open('result.txt','w') as f:
        f.write(result)
        print("评估报告已经保存为result.txt")
    return test_acc

# 4.可视化不同参数的表现
def plot_results(results):
    """可视化不同参数的表现"""
    metrics = list(set(r['metric'] for r in results))
    k_values = list(set(r['k'] for r in results))

    plt.figure(figsize=(10, 6))
    markers = {'euclidean': 'o', 'manhattan': 's', 'chebyshev': '^'}
    colors = {'euclidean': 'blue', 'manhattan': 'green', 'chebyshev': 'red'}

    for metric in metrics:
        metric_results = [r for r in results if r['metric'] == metric]
        accuracies = [r['dev_accuracy'] for r in metric_results]
        plt.plot(k_values, accuracies, marker=markers[metric],
                 color=colors[metric], label=metric)

    plt.title('Validation Accuracy for Different K and Distance Metrics')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.ylim(0.8, 1.0)  # 限定Y轴范围，更清晰展示差异
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('hyperparameter.png')
    print("超参数对比图已经保存为hyperparameter.png")
    plt.show()

if __name__ == '__main__':
    # 定义参数尝试
    k_values = [1,3,5,7,9,11]
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']  # 三种距离度量

    # 执行流程
    X_train, X_dev, X_test, y_train, y_dev, y_test, class_names = load_and_split()
    results, best_model = evaluate_knn_parameters(X_train, y_train, X_dev, y_dev, k_values, distance_metrics)
    evaluate_best_model(best_model, X_test, y_test)
    plot_results(results)

    # 保存模型
    with open('iris_knn_model.pkl', 'wb') as f:
        pickle.dump(best_model['model'], f)
        print("模型已保存!")
