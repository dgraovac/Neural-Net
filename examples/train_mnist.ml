(* Training a neural network on character recognition data *)

use "ml/neural_network/neural_network";
use "ml/neural_network/include/read_mnist.ml";

val training_images = "ml/neural_network/data/train-images-idx3-ubyte";
val training_labels = "ml/neural_network/data/train-labels-idx1-ubyte";

val testing_images = "ml/neural_network/data/t10k-images-idx3-ubyte";
val testing_labels = "ml/neural_network/data/t10k-labels-idx1-ubyte";

val training_data = ListPair.zip(produce_image_list (training_images), produce_label_list (training_labels));
val testing_data = ListPair.zip(produce_image_list (testing_images), produce_label_list (testing_labels));

fun perform_training net 0 = net
    | perform_training net n = perform_training (train_batch 0.3 training_data net) (n-1);

val network = (perform_training (create_network [784, 20, 10]) 4);

fun vec_output_to_int xs = get_index(xs, foldl Real.max 0.0 xs);

(* Testing the network *)
fun test network t_data =
    let fun count network ((inputs, outputs)::data) correct =
        if (vec_output_to_int ( feed_forward(inputs, network) )) = (vec_output_to_int ( outputs ))
        then count network data (correct+1)
        else count network data correct
        | count network [] correct = correct
    in count network t_data 0 end;

val proportion_correct = real(test network testing_data)/(real(get_length testing_data));
