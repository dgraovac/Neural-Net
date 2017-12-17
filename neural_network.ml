(* Basic feed forward neural network *)

use "ml/neural_network/include/useful_functions.ml";

datatype sigmoid_neuron = neuron of real list * real;


fun sigmoid_function(z) = 1.0/(1.0 + Math.exp(~z));

(* Does not apply the sigmoid function as values passed in have already had sigmoid_function applied *)
fun sig_prime(z) = z * (1.0 - z);


fun get_neuron_output (neuron(weights, bias)) inputs =
    sigmoid_function((dot weights inputs) + bias);

(* Derivative of quadratic cost function *)
fun cost_prime(outputs, expected_output) =
    zipWith(outputs, expected_output, (fn(x,y)=> Real.-(x,y)));


fun feed_forward(inputs, []) = inputs
    | feed_forward (inputs, layer::layers) =
        let fun eval_layer(inps, l::ls) =
            eval_layer((map (fn f => (f inps))) ((map get_neuron_output) l), ls)
            | eval_layer(inps, []) = inps
        in
        eval_layer(inputs, layer::layers)
        end;


fun feed_forward_get_activations (inputs, layer::layers) =
        let fun eval_layer(inps, l::ls) =
            let val current_outputs = ((map (fn f => (f inps))) ((map get_neuron_output) l))
            in
            (current_outputs)::eval_layer(current_outputs, ls)
            end
            | eval_layer(inps, []) = []
        in
        inputs::eval_layer(inputs, layer::layers)
        end;


fun create_weights 0 _ _ = []
    | create_weights n_inputs r_num_sequence max_size =
    (lazy_hd(r_num_sequence)/(Math.pow(real(max_size), 0.5)) )::(create_weights (n_inputs - 1) (lazy_tl(r_num_sequence)) max_size);


fun create_layer _ 0 = []
    | create_layer n_inputs n_nodes =
    neuron( (create_weights n_inputs (random_number 8074) n_inputs), lazy_hd(random_number 8074) )::create_layer n_inputs (n_nodes-1);


fun create_network (x::[]) = []
    | create_network (x::xs) = (create_layer x (hd(xs)))::create_network(xs)
    | create_network [] = raise network_creation_exception("Failure to create network");


fun update_layer ((neuron(weights, bias))::neurons) (delta_bias::dbs) (delta_weights::dws) =
    (neuron(zipWith(delta_weights, weights, (fn(x,y) => x+y)), bias + delta_bias))
    ::update_layer neurons dbs dws
    | update_layer _ _ _ = [];


fun back_sweep_deltas (neuron([], _)::neurons) _ = []
    | back_sweep_deltas neurons prev_deltas =
    ( dot ((map (fn (neuron(w::_, b)) => w | (neuron([], _)) => 0.0)) (neurons)) prev_deltas)::(back_sweep_deltas ((map (fn (neuron(w::ws, b)) => neuron(ws,b) | (neuron([], b)) => neuron([], b))) (neurons)) (prev_deltas));


fun calc_output_delta output expected_output =
    zipWith(map sig_prime output, cost_prime(output, expected_output), (fn (x,y) => x*y));


fun calc_hidden_delta last_layer last_deltas activations =
    zipWith(map sig_prime activations, back_sweep_deltas last_layer last_deltas, (fn (x,y) => x*y));


fun backpropagation is_output_layer (layer::layers) output expected_output eta (act::activations) prev_deltas last_layer =
    let
    val deltas = if is_output_layer
        then calc_output_delta output expected_output
        else calc_hidden_delta last_layer prev_deltas act
    in
    (update_layer layer ((map (fn x => ~x*eta)) (deltas))
    (map (fn d => (map (fn x => ~x*d*eta) (hd(activations)))) deltas))
    ::(backpropagation false layers output expected_output eta activations deltas layer)
    end
    | backpropagation _ [] _ _ _ _ _ _ = []
    | backpropagation _ _ _ _ _ _ _ _ = raise network_creation_exception("Network error: Number of activation layers found is incorrect.");


fun train inputs learning_rate network expected_output =
    backpropagation true (rev(network)) (feed_forward(inputs, network)) expected_output learning_rate (((rev((feed_forward_get_activations(inputs, network)))))) [] [];


fun train_batch eta [] net = net
    | train_batch eta ((inp, out)::dps) net = train_batch eta dps (rev(train inp eta net out));
