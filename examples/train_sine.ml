(* Training a neural network to the sine function *)

use "ml/neural_network.ml";

fun generate_data ~1 = []
    | generate_data n = ([real(n)/100.0], [Math.sin(real(n)/100.0)])::(generate_data (n-1));

val data = generate_data 157;

(* Will train the network n times with the whole dataset *)
fun perform_training net 0 = net
    | perform_training net n = perform_training (train_batch 0.01 data net) (n-1);

(* Creates a network and trains 10,000 times using the dataset. *)
val network = perform_training (create_network [1, 5, 1]) 30000;
