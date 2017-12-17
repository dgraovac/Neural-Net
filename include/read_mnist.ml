(* Reading and returning the MNIST dataset in a usable way *)

fun open_file path =
    let val stream = BinIO.openIn(path)
    fun loop ins =
    case BinIO.input1 ins of
        SOME line => line ::(loop ins)
        | NONE => []
in loop stream before BinIO.closeIn stream end;

fun data_to_real_list (x::xs) = (real(Word8.toInt(x))/255.0)::data_to_real_list(xs)
    | data_to_real_list [] = [];

fun data_to_int_list (x::xs) = Word8.toInt(x)::data_to_int_list(xs)
    | data_to_int_list [] = [];

fun produce_output_from_int k =
    let val init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    in take(init, k) @ [1.0] @ drop(init, k+1) end;

(* Reads the data, converts data to integers, removes first 8 items representing metadata *)
fun produce_label_list filename = map produce_output_from_int (drop(data_to_int_list(open_file(filename)), 8));

fun produce_image_list filename =
    let fun inner [] = []
        | inner xs = (take(xs, 784))::(inner (drop(xs, 784)))
    in inner (drop(data_to_real_list(open_file(filename)), 16)) end;
