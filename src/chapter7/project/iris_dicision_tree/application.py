import joblib
from sklearn.datasets import load_iris

# 加载物种名称（用于将数字标签转换为物种名称）
iris = load_iris()
target_names = iris.target_names

print("====鸢尾花分类模型/iris classification model====")
try:
    # 获取用户输入的特征
    print("(1/4)输入花萼长度(单位:cm)/input sepal length(unit cm):")
    sepal_length = float(input())

    print("(2/4)输入花萼宽度(单位:cm)/input sepal width(unit cm):")
    sepal_width = float(input())

    print("(3/4)输入花瓣长度(单位:cm)/input petal length(unit cm):")
    petal_length = float(input())

    print("(4/4)输入花瓣宽度(单位:cm)/input petal width(unit cm):")
    petal_width = float(input())

    print("=====正在计算中/computing...=====")

    # 加载模型并预测
    model = joblib.load('model.pkl')
    print("模型加载成功/Model loaded successfully!")

    # 准备输入特征（需要是二维数组格式）
    X = [[sepal_length, sepal_width, petal_length, petal_width]]

    # 预测类别和概率
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[0]  # 获取概率数组

    # 输出结果（转换为物种名称）
    print(f"\n预测的鸢尾花种类为/predicted type: {target_names[y_pred[0]]}")
    print("\n属于各品种的概率/possibilities for all types:")
    for name, prob in zip(target_names, y_proba):
        print(f"- {name}: {prob:.2%}")  # 格式化显示为百分比

except FileNotFoundError:
    print("错误: 未找到模型文件 'model.pkl'，请确认文件存在且路径正确。")
except ValueError:
    print("错误: 输入格式不正确，请确保输入的是数字。")
except Exception as e:
    print(f"发生错误: {str(e)}")
