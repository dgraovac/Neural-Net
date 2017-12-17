(* Useful functions for use in the neural network *)

fun zipWith(x::xs, y::ys, f) = f(x,y)::zipWith(xs, ys, f)
    | zipWith(_, _, _) = [];

fun dot xs ys = foldl op+ 0.0 (zipWith(xs, ys, (fn(x,y) => Real.*(x,y))));

fun rev [] = []
    | rev (x::xs) =
        let fun inner([], ys) = ys
            | inner(z::zs, ys) = inner (zs, z::ys)
    in inner(x::xs, []) end;

fun get_length xs = foldl op+ 0 (map (fn x => 1) xs);

fun get_index (xs, target) =
    let fun inner(z::zs, targ, n) =
        if Real.==(z, targ)
            then n
            else inner(zs, targ, n+1)
        |inner([], _, _) = 0
    in inner(xs, target, 0) end;

fun is_empty [] = true
    | is_empty _ = false;

fun hd (x::xs) = x;

fun tl (x::xs) = xs;

fun take (xs, 0) = []
    | take(x::xs, k) = x::take(xs, k-1);

fun drop (xs, 0) = xs
    | drop(x::xs, k) = drop(xs, k-1);

datatype 'a seq = Nil
    | Cons of 'a * (unit -> 'a seq);

fun lazy_tl (Cons(x,xf)) = xf();

fun lazy_hd (Cons(x,xf)) = x;


(* Stream of pseudorandom numbers using Blum Blum Shub generator*)
fun random_number k =
    let val next = (k*k) mod 16873
    in
    Cons((real(next)/16873.0), fn() => random_number(next))
    end;

exception network_creation_exception of string;
