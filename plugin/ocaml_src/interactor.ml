open Lymp

(* change "python3" to the name of your interpreter *)

let predict (x : int) : float =
	let interpreter = "/home/axel/miniconda3/envs/griffon/bin/python" in
	let py = init ~exec:interpreter "." in
	let simple = get_module py "python_src.simple" in
	let model = get_ref simple "MyNN" [Pyint 10] in
	let input = x in
	let prediction = get_float model "predict" [Pyint input] in
	prediction