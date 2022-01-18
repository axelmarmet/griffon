let __coq_plugin_name = "griffon_plugin"
let _ = Mltop.add_known_module __coq_plugin_name

# 9 "ocaml_src/griffon.mlg"
 
  (*** Dependencies from Coq ***)
  open Stdarg
  (* open Pp *)


let () = Vernacextend.vernac_extend ~command:"Predict" ~classifier:(fun _ -> Vernacextend.classify_as_query) ?entry:None 
         [(Vernacextend.TyML (false, Vernacextend.TyTerminal ("Predict", 
                                     Vernacextend.TyNonTerminal (Extend.TUentry (Genarg.get_arg_tag wit_int), 
                                     Vernacextend.TyNil)), (let coqpp_body i
                                                           () = Vernacextend.VtDefault (fun () -> 
                                                                
# 16 "ocaml_src/griffon.mlg"
                           
    let res = Interactor.predict i in
    Feedback.msg_notice (Pp.(++) (Pp.str "Predicted :") (Pp.real res))
  
                                                                ) in fun i
                                                           ~atts
                                                           -> coqpp_body i
                                                           (Attributes.unsupported_attributes atts)), None))]

