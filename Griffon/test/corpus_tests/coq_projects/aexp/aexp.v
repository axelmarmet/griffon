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

Theorem constant_fold_ok :
    forall ae,
    aeval(ae) = aeval(constant_fold(ae)).
Proof.
    induction ae.
    + simpl. reflexivity.
    + simpl. destruct (constant_fold ae1) eqn:E1;  destruct (constant_fold ae2) eqn:E2; simpl; auto.
    + simpl. destruct (constant_fold ae1) eqn:E1;  destruct (constant_fold ae2) eqn:E2; simpl; auto.
Qed.

