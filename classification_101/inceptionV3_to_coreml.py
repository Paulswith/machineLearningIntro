# coding: utf-8

import tensorflow as tf
import tfcoreml
from tensorflow.python.tools        import strip_unused_lib
from tensorflow.python.framework    import dtypes
from tensorflow.python.platform     import gfile



TF_MODEL_FILE     = '/Users/dobby/TEMP/SaveData/model/bazel_pd.pd'                 # 加载的pdtxt
FROZEN_MODEL_FILE = '/Users/dobby/TEMP/SaveData/model/prefix_pdmodel.pb'           # 将预处理好的保存下来.
COREML_MODEL_FILE = '/Users/dobby/TEMP/SaveData/model/inception_v3_labels.mlmodel'
Should_CHAN = False


## 1 加载图文件:
if Should_CHAN:
    with open(TF_MODEL_FILE, 'rb') as f:
        serialized = f.read()
    tf.reset_default_graph()
    original_gdef = tf.GraphDef()
    original_gdef.ParseFromString(serialized)

    ##2 可以在这里查看到全部的图信息(前提是你有转换成功)
    with tf.Graph().as_default() as g:
        tf.import_graph_def(original_gdef, name='')
        ops = g.get_operations()
        try:
            for i in range(10000):
                print('op id {} : op name: {}, op type: "{}"'.format(str(i),ops[i].name, ops[i].type))
        except:
            print("全部节点已打印完毕.")
            pass

    input_node_names = ['import/Mul', 'BottleneckInputPlaceholder']   # 本来以为是import/DecodeJpeg/contents, 实际上是Mul(tfcoreml-git上说的)
    output_node_names = ['import/pool_3/_reshape','final_train_ops/softMax_last']  # 想要保存的节点 , 'final_train_ops/softMax_last'

    gdef = strip_unused_lib.strip_unused(
            input_graph_def = original_gdef,
            input_node_names = input_node_names,
            output_node_names = output_node_names,
            placeholder_type_enum = dtypes.float32.as_datatype_enum)

    with gfile.GFile(FROZEN_MODEL_FILE, "wb") as f:
        f.write(gdef.SerializeToString())

print("开始进行模型转换:")
#############################################开始转换模型###################################################
input_tensor_shapes = {
                        "import/Mul:0":[1,299,299,3],          # batch size is 1
                        "BottleneckInputPlaceholder:0":[1,2048],
                       }
output_tensor_names = ['import/pool_3/_reshape:0','final_train_ops/softMax_last:0']

# Call the converter. This may take a while
coreml_model = tfcoreml.convert(
        tf_model_path=FROZEN_MODEL_FILE,
        mlmodel_path=COREML_MODEL_FILE,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names = ['import/Mul:0'],
        red_bias = -1,
        green_bias = -1,
        blue_bias = -1,
        image_scale = 2.0/255.0)
        # 必须调用进行图片的均值化



