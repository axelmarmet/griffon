open Proofview
open Environ
open Constr
open Context
open Sorts
open Util


let sexp_id (id : Names.Id.t) =
  ["("; "Id"; (Names.Id.to_string id); ")"]

let sexp_name (n : Names.Name.t) =
    match n with
    | Names.Name.Anonymous -> ["Anonymous"]
    | Names.Name id -> ["("; "Name"] @ (sexp_id id) @ [")"]

let sexp_modpath (m : Names.ModPath.t) =
  let sexp_dirpath (dp : Names.DirPath.t) =
      let inner_ids = ["("] @ List.flatten (List.map sexp_id (Names.DirPath.repr dp)) @ [")"] in
      ["(" ; "DirPath"] @ inner_ids @ [")"] in
  match m with
  | Names.MPfile dp -> ["(" ; "MPfile"] @ (sexp_dirpath dp) @ [")"]
  | _ -> ["Mistakeee"]


let sexp_kername (k : Names.KerName.t) =
  sexp_modpath (Names.KerName.modpath k) @
  ["(" ; "DirPath"; "(" ; ")"; ")"] @
  sexp_id (Names.Label.to_id (Names.KerName.label k))

let sexp_mutind (m : Names.MutInd.t) =
  ["(" ; "Mutind"] @
  (sexp_kername (Names.MutInd.user m)) @ [")"]

let sexp_mutind_int (t : Names.MutInd.t * int) =
  match t with
  | m, i -> "(" :: (sexp_mutind m) @ [string_of_int i ; ")"]

let sexp_univ (u : Univ.Instance.t) =
  ["(" ; "Instance" ; "(" ; ")" ; ")"]

let get_str (c : Constr.t) =
  let rec sexp_constr (c : Constr.t) =
    let inner = match kind c with
      | Rel i -> ["(" ; "Rel" ; string_of_int i ; ")"]
      | Prod (na, t, b) -> ["Prod"] @ (sexp_name na.binder_name) @ (sexp_constr t) @ (sexp_constr b)
      | Sort (s) ->
          let sort_string = match s with
            | Prop -> ["Prop"]
            | Set -> ["Set"]
            | _ -> ["Syke"]
          in
            "Sort" :: sort_string
      | App (b,l) -> ["App"] @ (sexp_constr b) @ ["("] @ List.flatten (List.map sexp_constr (Array.to_list l)) @ [")"]
      | Ind (t, u)-> ["Ind"; "("] @ (sexp_mutind_int t) @ (sexp_univ u) @ [")"]
      | _ -> []
    in
      "(" :: inner @ [")"]
  in
  String.concat " " (sexp_constr c)

let serialize_env () =
  Proofview.Goal.enter begin fun gl ->
    let hyps = Environ.named_context_val (Proofview.Goal.env gl) in
    (* giant hack *)
    let hyps_ctx = hyps.env_named_ctx in
      List.iter (fun x ->
        Feedback.msg_notice (Pp.str (get_str (Context.Named.Declaration.get_type x)))) hyps_ctx;
    tclUNIT ()
  end
