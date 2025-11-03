import os.path
import time

import tensorflow as tf
import keras
from keras import layers
from keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
import matplotlib.pyplot as plt

import numpy as np
import pathlib
import shutil
import tempfile

from numpy.f2py.crackfortran import verbose
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.models import save_model
from tensorflow_datasets.datasets.radon.radon_dataset_builder import features
from keras import mixed_precision
#下面的组合可以让矩阵乘法在Tensor Cores上运行
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)#开启XLA JIT
mixed_precision.set_global_policy('mixed_float16')#使用混合精度,可以显著降低训练时长


logdir = pathlib.Path(tempfile.mkdtemp())/'tensorboard_logs'
shutil.rmtree(logdir, ignore_errors=True)

#加载HIGGS数据集
dataset_dir = '/mnt/d/PycharmProjects/HIGGS/Datasets/'
zip_path = os.path.join(dataset_dir, 'HIGGS.csv.gz')
# print(zip_path)
file_path = keras.utils.get_file(
    fname= 'HIGGS.csv.gz',
    origin= 'file://' +zip_path,
    cache_dir= dataset_dir,
    extract=False
)

FEATURES = 28
#tf.data.experimental.CsvDataset 类可用于直接从 Gzip 文件读取 CSV 记录，而无需中间的解压步骤
ds = tf.data.experimental.CsvDataset(file_path,[float(),]*(FEATURES+1), compression_type='GZIP')
#CSV 读取器类会为每条记录返回一个标量列表。下面的函数会将此标量列表重新打包为 (feature_vector, label) 对
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], axis=1)
    label = tf.expand_dims(label,axis=-1)
    return features, label
#TensorFlow 在运算大批次数据时效率最高
#因此，不要单独重新打包每一行，而是创建一个新的 tf.data.Dataset，该数据集会接收以 10,000 个样本为单位的批次，将 pack_row 函数应用于每个批次，然后将批次重新拆分为单个记录
packer_ds = ds.batch(10000)\
    .map(pack_row)\
    .unbatch()

#检查这个新的 packed_ds 中的一些记录
for features, label in packer_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)

#为了减小训练篇幅，这里只使用前 1,000 个样本进行验证，再用接下来的 10,000 个样本进行训练
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
N_TEST = int(1e3)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 8192
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
AUTOTUNE = tf.data.AUTOTUNE




#切割数据集&构建管道

validate_ds = packer_ds.take(N_VALIDATION)
train_ds = packer_ds.skip(N_VALIDATION).take(N_TRAIN).cache('/mnt/d/PycharmProjects/HIGGS/cache_train')
test_ds = packer_ds.skip(N_VALIDATION+N_TRAIN).take(N_TEST)



validate_ds= validate_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).cache()
train_ds= train_ds.batch(BATCH_SIZE).shuffle(BUFFER_SIZE*4).repeat().prefetch(AUTOTUNE)
test_ds= test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

#如果在训练期间逐渐减小学习率，许多模型的训练效果会更好。请使用 tf.keras.optimizers.schedules 随着时间的推移减小学习率
lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False
)
def get_optimizer():
    return keras.optimizers.Adam(lr_schedule)
''''
上述代码设置了一个 tf.keras.optimizers.schedules.InverseTimeDecay，
用于在 1,000 个周期时将学习率根据双曲线的形状降至基础速率的 1/2，
在 2,000 个周期时将至 1/3，依此类推。
'''
step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize=(8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning rate')
plt.show()

#使用 callbacks.TensorBoard 为训练生成 TensorBoard 日志
def get_callbacks(name):
    # return [
    #     tfdocs.modeling.EpochDots(),
    #     keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    #     keras.callbacks.TensorBoard(logdir/name)
    # ]
    #优化以减少I/O压力
    log_dir = os.path.join(logdir, name)
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_binary_accuracy',
            patience=50,
            restore_best_weights=True,
            mode='max',
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir,
                                    update_freq='epoch',
                                    write_graph=False,
                                    write_images=False)
    ]
#对于大数据集或者复杂网络，我需要一个预估训练时间，这样就可以知道模型训练大概需要多少时间，然后我去忙其他事
def estimate_training_time(model, train_ds, steps_per_epoch,epochs=10, batch_size=500):
    """
       基于采样估算模型训练时间
       :param model: Keras 模型
       :param train_ds: 训练数据集
       :param steps_per_epoch: 每个 epoch 的步数
       :param epochs: 计划训练的 epoch 数
       :param batch_size: 批次大小
       :return: 预计总时间（秒）
       """
    print('正在预估训练时间（采样几步以估算总时长）...')

    #测试运行3个batch
    sample_steps = min(3,steps_per_epoch)
    #只取强sample_steps 个batch
    sample_ds = train_ds.take(sample_steps)


    start_time = time.time()
    #运行少量训练步骤（不保存权重）
    model.fit(sample_ds,
              steps_per_epoch=sample_steps,
              epochs=1,
              verbose=0)
    elapsed_time = time.time() - start_time
    avg_step_time = elapsed_time / sample_steps
    estimated_total_time = avg_step_time *steps_per_epoch*epochs
    print(f"平均每步耗时: {avg_step_time:.4f} 秒")
    print(f"预计总训练时间: {estimated_total_time / 60:.2f} 分钟 (约 {estimated_total_time / 3600:.2f} 小时)")

    return estimated_total_time

#每个模型将使用相同的 Model.compile 和 Model.fit 设置
def compile_and_fit(model, name, optimizer=None, max_epoches=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      keras.metrics.BinaryCrossentropy(
                          from_logits=True,name='binary_crossentropy'),
                      'accuracy'],
                  jit_compile=True)#Keras 默认是 Eager Execution，会逐步解释执行。可以开启图模式加速
    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epoches,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)
    return history

# #微模型
# tiny_model = keras.Sequential([
#     layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
#     layers.Dense(1)
# ])
# size_histories = {}
# size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

#小模型
small_model = keras.Sequential([
    layers.Dense(16,activation='elu',kernel_regularizer=regularizers.l2(0.0001),
                 input_shape=(FEATURES,)),
    layers.Dropout(0.2),
    layers.Dense(16,activation='elu',kernel_regularizer=regularizers.l2(0.0001),),
    layers.Dropout(0.2),
    layers.Dense(1)
])
small_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy'],
    jit_compile=True #Keras 默认是 Eager Execution，会逐步解释执行。可以开启图模式加速
)
_= estimate_training_time(small_model,train_ds,STEPS_PER_EPOCH, epochs=10000,batch_size=BATCH_SIZE)
size_histories = {}
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')
#查看模型的表现
plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.show()

#评估模型
results = small_model.evaluate(test_ds, return_dict=True)
print('测试集性能')
for key, value in results.items():
    print(f'{key}: {value}')

#保存模型
small_model.save('higgs_small.keras')