let __coq_plugin_name = "griffon_plugin"
let _ = Mltop.add_known_module __coq_plugin_name

# 9 "ocaml_src/griffon.mlg"
 
  (*** Dependencies from Coq ***)
  open Stdarg
  open Ltac_plugin
  (* open Pp *)


let () = Vernacextend.vernac_extend ~command:"predict" ~classifier:(fun _ -> Vernacextend.classify_as_query) ?entry:None 
         [(Vernacextend.TyML (false, Vernacextend.TyTerminal ("Predict", 
                                     Vernacextend.TyNonTerminal (Extend.TUentry (Genarg.get_arg_tag wit_int), 
                                     Vernacextend.TyNil)), (let coqpp_body i
                                                           () = Vernacextend.VtDefault (fun () -> 
                                                                
# 17 "ocaml_src/griffon.mlg"
                           
    let res = Interactor.predict i in
    Feedback.msg_notice (Pp.(++) (Pp.str "Predicted :") (Pp.real res))
  
                                                                ) in fun i
                                                           ~atts
                                                           -> coqpp_body i
                                                           (Attributes.unsupported_attributes atts)), None))]

let () = Tacentries.tactic_extend __coq_plugin_name "ser_env" ~level:0 
         [(Tacentries.TyML (Tacentries.TyIdent ("Serialize", Tacentries.TyNil), 
           (fun ist -> 
# 25 "ocaml_src/griffon.mlg"
    Hypserializer.serialize_env () 
           )))]

let () = Tacentries.tactic_extend __coq_plugin_name "log_env" ~level:0 
         [(Tacentries.TyML (Tacentries.TyIdent ("Log", Tacentries.TyArg (
                                                       Extend.TUentry (Genarg.get_arg_tag wit_string), 
                                                       Tacentries.TyNil)), 
           (fun s ist -> 
# 30 "ocaml_src/griffon.mlg"
    Hypserializer.log_env (s) 
           )))]

