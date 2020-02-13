import cv2
import paddle.fluid as fluid
from image_utils import transform_img
from vggnet import VGG


# 定义预测方法
def pred(model, params_file_path, test_picture):
    with fluid.dygraph.guard():
        img = cv2.imread(test_picture)
        img = transform_img(img)
        img = img.reshape(1, 3, 224, 224)
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()
        img = fluid.dygraph.to_variable(img)
        logits = model(img)
        result = fluid.layers.sigmoid(logits).numpy()
        print("result is: {}".format(result))
        score = result[0][0]
        label = 1 if score >= 0.5 else 0
        label_name = 'dog' if score >= 0.5 else 'cat'
        print("test picture is [{}]{}, the score is {}".format(label, label_name, score))


if __name__ == '__main__':
    param_file_path = 'dogs_vs_cats_9'
    # test_picture = 'dogs-vs-cats-redux-kernels-edition/test/9026.jpg'
    # test_picture = 'D:\\temp\\dog1.jpg'
    # test_picture = 'D:\\temp\\cat1.jpg'
    test_picture = 'D:\\temp\\charley.png'
    with fluid.dygraph.guard():
        model = VGG('vgg')
    pred(model, param_file_path, test_picture)
