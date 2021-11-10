Theorem evar_check :
  forall
    (R : nat -> nat -> Prop),
    (forall x y z, R x y -> R y z -> R x z) ->
    (forall n m p, R n m /\ R m p -> R n p).
Proof.
  intros; destruct H0 as [H0 H1].
  eapply H.
  * apply H0.
  * apply H1.
Qed.


Theorem negation_fn_applied_twice :
  forall (f : bool -> bool),
  (forall (x : bool), f x = negb x) ->
  forall (b : bool), f (f b) = b.
Proof.
  intros.
  rewrite H. rewrite H.
  case b; reflexivity.
Qed.
