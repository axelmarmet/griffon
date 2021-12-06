(* Type inference *)
Theorem app_type_1:
(*  forall (A B : Type) (b : B) (x : A), exists (f : A -> mask),  f x = b. *)
    forall (A B : Type) (b : B) (x : A), exists (f : A -> B),  f x = b.
Proof.
    intros. exists (fun a : A => b). reflexivity.
Qed.

Theorem app_type_2:
(*  forall (A B : Type) (b : B) (x : A), exists (f : mask -> B),  f x = b. *)
    forall (A B : Type) (b : B) (x : A), exists (f : A -> B),  f x = b.
Proof.
    intros. exists (fun a : A => b). reflexivity.
Qed.

Theorem app_type_3:
(*  forall (A B : Type) (b : B) (x : mask), exists (f : A -> B),  f x = b. *)
    forall (A B : Type) (b : B) (x : A), exists (f : A -> B),  f x = b.
Proof.
    intros. exists (fun a : A => b). reflexivity.
Qed.

Theorem app_type_4:
(*  forall (A B : Type) (b : mask) (x : A), exists (f : A -> B),  f x = b. *)
    forall (A B : Type) (b : B) (x : A), exists (f : A -> B),  f x = b.
Proof.
    intros. exists (fun a : A => b). reflexivity.
Qed.
