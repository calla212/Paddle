import paddle.v2 as paddle


def softmax_regression(img):
    predict = paddle.layer.fc(input=img,
                              size=10,
                              act=paddle.activation.Softmax())
    return predict


def multilayer_perceptron(img):
    # The first fully-connected layer
    hidden1 = paddle.layer.fc(input=img, size=128, act=paddle.activation.Relu())
    # The second fully-connected layer and the according activation function
    hidden2 = paddle.layer.fc(input=hidden1,
                              size=64,
                              act=paddle.activation.Relu())
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    predict = paddle.layer.fc(input=hidden2,
                              size=10,
                              act=paddle.activation.Softmax())
    return predict


def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Tanh())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Tanh())
    # The first fully-connected layer
    fc1 = paddle.layer.fc(input=conv_pool_2,
                          size=128,
                          act=paddle.activation.Tanh())
    # The softmax layer, note that the hidden size should be 10,
    # which is the number of unique digits
    predict = paddle.layer.fc(input=fc1,
                              size=10,
                              act=paddle.activation.Softmax())
    return predict


def main():
    paddle.init(use_gpu=True, trainer_count=1)

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))

    predict = softmax_regression(images)
    #predict = multilayer_perceptron(images)
    #predict = convolutional_neural_network(images)

    cost = paddle.layer.classification_cost(input=predict, label=label)

    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1 / 128.0,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                result = trainer.test(reader=paddle.reader.batched(
                    paddle.dataset.mnist.test(), batch_size=128))
                print "Pass %d, Batch %d, Cost %f, %s, Testing metrics %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics,
                    result.metrics)

    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=128),
        event_handler=event_handler,
        num_passes=100)


if __name__ == '__main__':
    main()
