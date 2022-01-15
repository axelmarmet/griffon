open Lymp

(* change "python3" to the name of your interpreter *)
let interpreter = "python3"
let py = init ~exec:interpreter "."
let simple = get_module py "python_src.simple"

let () =
	let model = get_ref simple "MyNN" [Pyint 10] in
	let input = 1 in
	let prediction = get_float model "predict" [Pyint input] in
	Printf.printf "f(%d) = %f" input prediction ;
	close py