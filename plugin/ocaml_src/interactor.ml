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

let convert_statements (statements : (string * string) list) : pyobj =
	Pylist (List.map (fun (id,sexpr) -> Pytuple [Pystr id ; Pystr sexpr]) statements)

let interact (statements : (string * string) list) : string =
	let pobj = convert_statements statements in
	let interpreter = "/home/axel/miniconda3/envs/griffon/bin/python" in
	let py = init ~exec:interpreter "." in
	let simple = get_module py "griffon.preprocessing.inference" in
	get_string simple "get_statement_batch" [pobj]

let log (title : string) (statements : (string * string) list) : unit =
	let pobj = convert_statements statements in
	let interpreter = "/home/axel/miniconda3/envs/griffon/bin/python" in
	let py = init ~exec:interpreter "." in
	let simple = get_module py "griffon.preprocessing.inference" in
	ignore (get simple "log_statements" [Pystr title ; pobj])