import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define a couple of hyperparameters
input_size = 28*28
hidden_size = 100
output_size = 10
batch_size = 100
num_epochs = 10

# Build the TF computation graph

class ShallowVanillaNN(object):
    """
    This is an augumented implementation of the MNIST tutorial code from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
    """
    def __init__(self, is_training=True):
        self.is_training = is_training

        # Placeholders for the input data (xs) and labels (ys)
        self.xs = tf.placeholder(tf.float32, [None, input_size])
        self.ys = tf.placeholder(tf.float32, [None, output_size])

        # Hidden layer
        hidden_weight = tf.Variable(tf.random_normal([input_size, hidden_size]))
        hidden_bias = tf.Variable(tf.random_normal([hidden_size]))

        hidden_layer = tf.matmul(self.xs, hidden_weight) + hidden_bias
        tf.summary.histogram('pre_activations', hidden_layer)
        hidden_layer = tf.nn.tanh(hidden_layer)
        tf.summary.histogram('activations', hidden_layer)

        # Output layer
        output_weight = tf.Variable(tf.random_normal([hidden_size, output_size]))
        output_bias = tf.Variable(tf.random_normal([output_size]))

        output_layer = tf.matmul(hidden_layer, output_weight) + output_bias

        if is_training:
            self.loss = tf.reduce_mean( \
                tf.nn.softmax_cross_entropy_with_logits(output_layer, self.ys))

            # This is for logging in TensorBoard
            tf.summary.scalar("Loss", self.loss)

            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(self.ys,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            self.prediction = tf.nn.softmax(output_layer)

        self.merged_summ = tf.summary.merge_all()

    def get_loss(self):
        if not self.is_training:
            raise Error("The model is not in training mode right now!")
        return self.loss

    def get_train_step(self):
        if not self.is_training:
            raise Error("The model is not in training mode right now!")
        return self.train_step

    def get_train_accuracy(self):
        if not self.is_training:
            raise Error("The model is not in training mode right now!")
        return self.accuracy

    def get_pred(self):
        if not self.is_training:
            raise Error("The model is not in prediction mode right now!")
        return self.prediction

    def get_summary(self):
        return self.merged_summ


# Run the graph

def run():

    shallow_net = ShallowVanillaNN()
    loss = shallow_net.get_loss()
    optimiser = shallow_net.get_train_step()
    accuracy = shallow_net.get_train_accuracy()
    summary = shallow_net.get_summary()

    with tf.Session() as sesh:

        train_writer = tf.train.SummaryWriter('./train',
                                          sesh.graph)

        init = tf.global_variables_initializer()

        sesh.run(init)

        summary_counter = 0

        for epoch in range(num_epochs):
            print "Epoch", (epoch+1), "started..."
            num_batches = mnist_data.train.num_examples // batch_size
            for i in range(num_batches):
                train_data, labels = mnist_data.train.next_batch(batch_size)
                _, batch_loss, summ = sesh.run([optimiser, loss, summary], feed_dict={shallow_net.xs: train_data, shallow_net.ys: labels})
                #print "Batch", (i+1), "of", num_batches, "finished. Loss:", batch_loss

                if summary_counter % 10:
                    train_writer.add_summary(summ, summary_counter//10)

                summary_counter += 1

            print "Epoch", (epoch+1), "completed."

        acc = sesh.run(accuracy, feed_dict={shallow_net.xs: mnist_data.test.images, shallow_net.ys: mnist_data.test.labels})
        print "Final accuracy:", acc

if __name__ == '__main__':
    run()
