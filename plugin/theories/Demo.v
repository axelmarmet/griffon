From Griffon Require Import Loader.

Predict 23.


Theorem conj_intro:
(*  forall (W X Y Z : Prop), (W -> X) /\ (Y -> Z) -> (W /\ Y -> mask /\ mask) *)
    (forall (W X Y Z : Prop), (W -> X) /\ (Y -> Z) -> (W /\ Y -> X /\ Z)) -> True.
Proof.
    intros.
    Serialize.
    intros.
    Serialize.
    intuition.
Qed.

Inductive aexp: Set :=
| AConst  : nat -> aexp
| APlus   : aexp -> aexp -> aexp
| AMult   : aexp -> aexp -> aexp.

Fixpoint aeval (ae : aexp) : nat :=
    match ae with
    | AConst n => n
    | APlus ae1 ae2 => aeval(ae1) + aeval(ae2)
    | AMult ae1 ae2 => aeval(ae1) * aeval(ae2)
    end.

Fixpoint constant_fold (ae : aexp) : aexp :=
    match ae with
    | APlus ae1 ae2 =>
        let oae1 := constant_fold ae1 in
        let oae2 := constant_fold ae2 in
        match oae1, oae2 with
        | AConst a, AConst b => AConst (a + b)
        | _,_ => APlus oae1 oae2
        end
    | AMult ae1 ae2 =>
        let oae1 := constant_fold ae1 in
        let oae2 := constant_fold ae2 in
        match oae1, oae2 with
        | AConst a, AConst b => AConst (a * b)
        | _,_ => AMult oae1 oae2
        end
    | _ => ae
    end.

Fixpoint algebraic_simpl (ae : aexp) : aexp :=
    match ae with
    | APlus (AConst 0) ae2 => ae2
    | APlus ae1 (AConst 0) => ae1
    | AMult (AConst 0) _ => AConst 0
    | AMult _ (AConst 0) => AConst 0
    | AMult (AConst 1) ae2 => ae2
    | AMult ae1 (AConst 1) => ae1
    | _ => ae
    end.

Theorem comp_ok :
    forall aex, aeval (constant_fold (algebraic_simpl aex)) = aeval aex.
Proof.
    intros.
    Serialize.
Admitted.



