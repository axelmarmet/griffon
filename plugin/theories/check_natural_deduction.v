(* Natural deduction *)
From Griffon Require Import Loader.

Theorem assumption:
(*  forall (P : Prop), P -> mask *)
    forall (P : Prop), P -> P.
Proof.
    Log "assumption".
    auto.
Qed.

Theorem conj_intro:
(*  forall (W X Y Z : Prop), (W -> X) /\ (Y -> Z) -> (W /\ Y -> mask /\ mask) *)
    forall (W X Y Z : Prop), (W -> X) /\ (Y -> Z) -> (W /\ Y -> X /\ Z).
Proof.
    Log "conj_intro".
    intuition.
Qed.

Theorem conj_elim_left:
(*  forall (X Y Z : Prop), (mask /\ Y) -> (X)  *)
    forall (X Y Z : Prop), (X /\ Y) -> (X).
Proof.
    Log "conj_elim_left".
    intuition.
Qed.

Theorem conj_elim_right:
(*  forall (X Y Z : Prop), (X /\ mask) -> (Y)  *)
    forall (X Y Z : Prop), (X /\ Y) -> (Y).
Proof.
    Log "conj_elim_right".
    intuition.
Qed.

Theorem disj_intro_left:
(*  forall (X Y : Prop), X -> (mask \/ Y). *)
    forall (X Y : Prop), X -> (X \/ Y).
Proof.
    Log "disj_intro_left".
    intuition.
Qed.

Theorem disj_intro_right:
(*  forall (X Y : Prop), Y -> (X \/ mask). *)
    forall (X Y : Prop), Y -> (X \/ Y).
Proof.
    Log "disj_intro_right".
    intuition.
Qed.

Theorem disj_elim:
(*  forall (X Y Z : Prop), (X \/ Y) -> (X -> mask) -> (Y -> mask) -> Z. *)
    forall (X Y Z : Prop), (X \/ Y) -> (X -> Z) -> (Y -> Z) -> Z.
Proof.
    Log "disj_elim".
    intuition.
Qed.

Theorem impl_elim:
(*  forall (P Q : Prop), P -> (P -> mask) -> Q. *)
    forall (P Q : Prop), P -> (P -> Q) -> Q.
Proof.
    Log "impl_elim".
    intuition.
Qed.

Theorem neg_elim:
(*  forall (P : Prop), P -> ~P -> mask. *)
    forall (P : Prop), P -> ~P -> False.
Proof.
    Log "neg_elim".
    intuition.
Qed.

Theorem aff:
(*  forall (P Q : Prop), mask -> (Q -> P). *)
    forall (P Q : Prop), P -> (Q -> P).
Proof.
    Log "aff".
    intuition.
Qed.

Theorem contract:
(*  forall (P :Prop), P /\ P -> mask.     *)
    forall (P :Prop), P /\ P -> P.
Proof.
    Log "contract".
    intuition.
Qed.