import tensorflow as tf
import pickle

n_nodes_h1 = 64

n_classes = 2
batch_size = 10

x = tf.placeholder("float", [None, 4096])
y = tf.placeholder("float")


def neural_net(data):
    hidden_layer_1 = {"weights": tf.Variable(tf.random_normal([4096, n_nodes_h1])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_h1]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_h1, n_classes])),
                    "biases": tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1["weights"]), hidden_layer_1["biases"])
    l1 = tf.nn.relu(l1)

    output = tf.add(tf.matmul(l1, output_layer["weights"]), output_layer["biases"])

    return output


def train_neural_net(x):
    prediction = neural_net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    no_of_epochs = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start = 0
        end = start + batch_size

        for epoch in range(no_of_epochs):
            epoch_loss = 0

            for i in range(int(len(x_train_data)/batch_size)):
                epoch_x = x_train_data[start:end]
                epoch_y = y_train_data[start:end]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                end = end + batch_size

            print("Epoch ", epoch, " Completed out of ", no_of_epochs, " Loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy ", accuracy.eval({x: x_test_data, y: y_test_data}))


def get_training_xy():
    pickle_file = open("training_obj.pickle", "rb")
    data = pickle.load(pickle_file)
    x = []
    y = []
    for example in data:
        x.append(example["features"])

    for answers in data:
        y.append([answers["label"]])

    return x, y


def get_test_xy():
    pickle_file = open("test_obj.pickle", "rb")
    data = pickle.load(pickle_file)
    x = []
    y = []
    for example in data:
        x.append(example["features"])

    for answers in data:
        y.append([answers["label"]])

    return x, y


x_train_data, y_train_data = get_training_xy()
x_test_data, y_test_data = get_test_xy()


train_neural_net(x)
