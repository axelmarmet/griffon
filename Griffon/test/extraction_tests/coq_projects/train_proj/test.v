Require Import List.
Import ListNotations.

CoInductive Stream : Set := Seq : nat -> Stream -> Stream.

CoFixpoint from (n:nat) : Stream := Seq n (from (S n)).

Fixpoint approx (s : Stream) (n : nat) : list nat :=
  match n with
    | O => nil
    | S n' =>
      match s with
        | Seq h t => h :: approx t n'
      end
  end.

Fixpoint from_list (n number_tokens : nat) : list nat :=
    match number_tokens with
    | O => []
    | S number_tokens' => n :: from_list (S n) number_tokens'
    end.

Theorem stupid :
  forall s n, from_list s n = [] ++ from_list s n.
Proof.
  reflexivity.
Qed.

Theorem test_cofix :
  forall s n, from_list s n = approx (from s) n.
Proof.
  intros.
  unfold from.
  rewrite stupid. simpl.
  generalize dependent s.
  induction n; intros.
  * simpl. reflexivity.
  * simpl. rewrite <- IHn. reflexivity.
Qed.

Definition is_empty {A : Type} (l:list A) : bool :=
match l with
  | [] => true
  | x :: l' => false
end.


Parameters A B : Set.

Inductive tree : Set := node : A -> forest -> tree

with forest : Set :=
| leaf : B -> forest
| cons : tree -> forest -> forest.

Fixpoint tree_size (t:tree) : nat :=
match t with
| node a f => S (forest_size f)
end
with forest_size (f:forest) : nat :=
match f with
| leaf b => 1
| cons t f' => (tree_size t + forest_size f')
end.

Theorem multi_fix_thm :
  forall t : tree, 0 <= tree_size(t).
Proof.
  intros.
  unfold tree_size.
  apply le_0_n.
Qed.


Theorem vacuous_match :
  forall (A : Type) (l1 : list A) , andb (is_empty l1) false = false.
Proof.
  intros.
  unfold is_empty.
  apply Bool.andb_false_r.
Qed.


Theorem obvious :
  (let x := 2 in x + x) = 4.
Proof.
  reflexivity.
Qed.

Theorem other_obvious :
  (let x := 2 in x + x) = 4.
Proof.
  apply obvious.
Qed.

Theorem fix_test :
  forall (A : Type) (l1 : list A), 0 <= length l1.
Proof.
  intros.
  unfold length.
  apply PeanoNat.Nat.le_0_l.
Qed.

Theorem de_morgan:
    forall A B : Prop,
    (~ A) \/ (~ B) -> ~ (A /\ B).
Proof.
    unfold not in *.
    intros.
    destruct H as [H | H].
    * destruct H0.
      apply H. apply H0.
    * destruct H0.
      apply H. apply H1.
Qed.