Theorem add_0_right :
    forall n : nat, n + 0 = n.
Proof.
    induction n; try reflexivity.
    auto.
Qed.

Theorem commut :
    forall a b : nat, a + b = b + a.
Proof.
    induction a; intros.
    + simpl. rewrite add_0_right.
      reflexivity.
    + induction b.
        - simpl. rewrite add_0_right.
          reflexivity.
        - simpl. rewrite IHa. rewrite <- IHb. simpl. rewrite IHa. reflexivity.
Qed.

Theorem weird_com :
    forall a b : nat, (a : nat) + b = b + a.
Proof.
    intros.
    apply commut.
Qed.

Theorem lambda_test :
    forall n : nat, (fun a : nat => a + 1) n = 1 + n.
Proof.
    intro.
    rewrite <- commut.
    reflexivity.
Qed.

Inductive ev : nat -> Prop :=
| ev_0 : ev 0
| ev_SS (n : nat) (H : ev n) : ev (S (S n)).

Fixpoint double(n : nat) :=
match n with
| 0 => 0
| S n' => S (S (double n'))
end.

Theorem double_is_ev :
    forall n, ev (double n).
Proof.
    induction n.
    * simpl. apply ev_0.
    * simpl. apply ev_SS. apply IHn.
Qed.