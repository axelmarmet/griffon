(* Inductive day : Type :=
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday.

Definition next_weekday (d:day) : day :=
  match d with
  | monday    => tuesday
  | tuesday   => wednesday
  | wednesday => thursday
  | thursday  => friday
  | friday    => monday
  | saturday  => monday
  | sunday    => monday
  end.

Example test_next_weekday:
  (next_weekday (next_weekday saturday)) = tuesday.
Proof. simpl. reflexivity.  Qed. *)

Inductive ev : nat -> Prop :=
| ev_0 : ev 0
| ev_SS (n : nat) (H : ev n) : ev (S (S n)).

Fixpoint double (n : nat) : nat :=
  match n with
  | 0 => 0
  | S n' => S (S (double n'))
  end.

Theorem ev_double : forall n, ev (double n).
Proof.
  intros.
  induction n.
  * simpl. apply ev_0.
  * simpl. Search (ev (S (S _))). apply ev_SS. apply IHn.
Qed.