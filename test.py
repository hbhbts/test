from matplotlib import pyplot
import numpy as np
from caffe2.python import model_helper, workspace, brew, core

print("LeNet\n")


# data input

def AddInput(model, batch_size, db, db_type):
    data_uint8, label = model.TensorProtosDBInput(
        [],
	["data_unit8", "label"],
	batch_size=batch_size,
	db=db,
	db_type=db_type)

    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    data = model.Scale(data, data, scale=float(1.0/256))
    data = model.StopGradient(data, data)
    return data, label


def AddNet(model, data):
    # image size: 28x28 -> 24x24
    conv1 = brew.conv(model, data, "conv1", dim_in=1, dim_out=20, kernel=5, stride=1)
    # image size: 24x24 -> 12x12
    pool1 = brew.max_pool(model, conv1, "pool1", kernel=2, stride=2)
    # image size: 12x12 -> 8x8
    conv2 = brew.conv(model, pool1, "conv2", dim_in=20, dim_out=100, kernel=5, stride=1)
    # image size: 8x8 -> 4x4
    pool2 = brew.max_pool(model, conv2, "pool2", kernel=2, stride=2)
    # image size: 100x4x4 -> 500
    fc3 = brew.fc(model, pool2, "fc3", dim_in=100*4*4, dim_out=500)
    relu = brew.relu(model, fc3, fc3)
    # image size: 500 -> 10
    fc4 = brew.fc(model, fc3, "fc4", dim_in=500, dim_out=10)
    # softmax
    softmax = brew.softmax(model, fc4, "softmax")
    return softmax

def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")

def AddTrainNet(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], "xent")
    loss = model.AveragedLoss(xent, "loss")

    model.AddGradientOperators([loss])

    ITER = brew.iter(model, "iter")
    LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999)

    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

    for param in model.params:
        param_grad = model.param_to_grad[param]


def UserCreateNet(train_model, test_model):
    data, label = AddInput(train_model, 64, "./mnist-train-nchw-lmdb", "lmdb")
    softmax = AddNet(train_model, data)

    #accuracy
    AddAccuracy(train_model, softmax, label)

    AddTrainNet(train_model, softmax, label)
    # print(str(model.param_init_net.Proto()) + "\n...")

    # parameter initialization
    workspace.RunNetOnce(train_model.param_init_net)
    # print("workspace blobs:".format(workspace.Blobs()))

    # creating the network
    workspace.CreateNet(train_model.net, overwrite=False)

    data, label = AddInput(
            test_model,
            batch_size=100,
            db="./mnist-test-nchw-lmdb",
            db_type="lmdb")
    softmax = AddNet(test_model, data)
    AddAccuracy(test_model, softmax, label)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)


def UserRunNet(N, train_model, test_model, train_iters, test_iters):
    for i in range(train_iters):
        workspace.RunNet(train_model.net)

    test_accuracy = np.zeros(test_iters)
    for i in range(test_iters):
        workspace.RunNet(test_model.net.Proto().name)
        test_accuracy[i] = workspace.FetchBlob("accuracy")

    print(test_accuracy)
    print("ITER INDEX: {}\t\t\ttest_accuracy={}\n".format(N, test_accuracy.mean()))


train_model = model_helper.ModelHelper(
            name="mnist_train",
            arg_scope={"order": "NCHW"})

test_model = model_helper.ModelHelper(
            name="mnist_test",
            arg_scope={"order": "NCHW"},
            init_params=False)

UserCreateNet(train_model, test_model)

N = 100
for i in range(N):
    UserRunNet(i, train_model, test_model, 100, 1)


#pyplot.plot(loss, 'b')
#pyplot.plot(accuracy, 'r')
#pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
#pyplot.show()



