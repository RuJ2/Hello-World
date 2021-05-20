from flask import Flask, render_template, request, flash
import sqlite3
import json
import numpy as np
import pandas as pd
from Regresser import ANN, Ensemble
from saga import GA

ann = ANN()
ann.load_model()
ensemble = Ensemble()
ensemble.load_model()

app = Flask(__name__)
app.secret_key = 'sec123'

INPUT_NUM = 17
OUTPUT_NUM = 1


@app.route('/')
def hello():
    # 默认返回主页面 
    print("加载主页面···")
    return render_template('主页.html')


@app.route('/predict')
def predict():
    # 预测页面绘图
    print("加载预测页面···")

    data_pd = pd.read_csv('data/weather.csv').values[-15:, :]  # 取连续的15组数据
    date = list(data_pd[:, 0])  # 日期
    x = data_pd[:, 1:-OUTPUT_NUM]  # 变量值
    y_tst = list(data_pd[:, -1])  # 样本结果值
    y_ann = list(ann.predict(x))
    # y_ens = list(ensemble.predict(x))
    y_ens = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    Content = {'data': [y_tst, y_ann, y_ens],
               'legends': ["Samples", 'ANN_Predict', 'Ensemble_Pre'],
               'len': 3,
               'date': date
               }
    return render_template('predict.html', **Content)


@app.route('/train', methods=['GET', 'POST'])
def train():
    print("加载训练页面···")

    with open('db/train_data.json') as file_obj:
        content = json.load(file_obj)
        print("加载页面表单数据成功")

    if request.method == 'POST':
        # 当点击提交获取页面表单数据
        content = {'model': request.form.get("model"),
                   'minTrainSize': request.form.get("minTrainSize"),
                   'batchSize': request.form.get("batchSize"),
                   'trainCoff': request.form.get("trainCoff"),
                   'dateStart': request.form.get("dateStart"),
                   'dateEnd': request.form.get("dateEnd")}

        if content['dateStart'] >= content['dateEnd']:
            flash("日期输入有误，起始日期更大！")

        else:
            # 模型训练
            print("模型训练:", content["model"])
            if content["model"] == "ann":
                ann.update_model(int(content["batchSize"]))
            elif content["model"] == "ensemble":
                ensemble.update_model(int(content["batchSize"]))

            # 保存数据
            with open('db/train_data.json', 'w') as file_obj:
                json.dump(content, file_obj)
                print("保存json成功！")
            flash("模型训练成功！")
    return render_template('train.html', **content)


@app.route('/ga', methods=['GET', 'POST'])
def saga():
    show_result = False

    print("加载ga页面···")
    with open('db/ga_data.json') as file_obj:
        content = json.load(file_obj)

    if request.method == 'POST':
        # 当点击提交获取页面表单数据
        content = {'model': request.form.get("model"),
                   'pop_size': request.form.get("pop_size"),
                   'chrome_len': request.form.get("chrome_len"),
                   'pm': request.form.get("pm"),
                   'pc': request.form.get("pc")}
        # 模型训练
        # 保存数据
        with open('db/ga_data.json', 'w') as file_obj:
            json.dump(content, file_obj)
            print("保存json成功！")
        flash("获取参数成功！")
        show_result = True

    def f(x):
        return ann.predict(np.array(x).reshape(1,-1))[0]

    pop_size = 100
    num = INPUT_NUM
    bound = [[0 for i in range(INPUT_NUM)], [10 for i in range(INPUT_NUM)]]
    chromosome_length = 10
    pc = 0.6
    pm = 0.5

    # content = {'model': "ann",
    #            'pop_size': 100,
    #            'chrome_len': 20,
    #            'pm': 0.5,
    #            'pc': 0.6}

    ga = GA(pop_size, chromosome_length, num, bound, pc, pm, f)
    result = ga.solver()

    res = {
        "max_value": result[0],
        "param": result[1],
        "show_result": show_result
    }
    return render_template('ga.html', **content, **res)


if __name__ == '__main__':
    print("运行main···")
    app.run(port=5000, debug=True)
