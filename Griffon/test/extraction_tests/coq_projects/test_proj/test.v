Theorem find_in_hypo:
  forall P Q : Prop,
  P -> Q -> P /\ Q.
Proof.
  intros P Q H1 H2.
  split.
  + apply H1.
  + apply H2.
Qed.

Theorem right_zero_neutral :
  forall n,
  n + 0 = n.
Proof.
  induction n.
  + reflexivity.
  + simpl. rewrite IHn. reflexivity.
Qed.

Theorem commutative :
  forall a b : nat,
  a + b = b + a.
Proof.
  induction a; intros.
  + simpl. symmetry. apply right_zero_neutral.
  + induction b.
    - simpl. rewrite IHa. simpl. reflexivity.
    - simpl. rewrite <- IHb. simpl. rewrite IHa. simpl. rewrite IHa. reflexivity.
Qed.
