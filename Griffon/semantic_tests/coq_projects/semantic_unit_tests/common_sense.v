Require Import Arith.

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
    | APlus ae1 ae2 =>
        let oae1 := algebraic_simpl ae1 in
        let oae2 := algebraic_simpl ae2 in
        match oae1, oae2 with
        | AConst 0, _ => oae2
        | _, AConst 0 => oae1
        | _,_ => APlus oae1 oae2
        end
    | AMult ae1 ae2 =>
        let oae1 := algebraic_simpl ae1 in
        let oae2 := algebraic_simpl ae2 in
        match oae1, oae2 with
        | AConst 0, _ => AConst 0
        | _, AConst 0 => AConst 0
        | AConst 1, _ => oae2
        | _, AConst 1 => oae1
        | _,_ => AMult oae1 oae2
        end
    | _ => ae
    end.

Theorem constant_fold_ok :
    forall aex, aeval (constant_fold aex) = aeval aex.
Proof.
    induction aex; simpl.
    + reflexivity.
    + destruct (constant_fold aex1); destruct (constant_fold aex2); simpl in *;
      rewrite IHaex1; rewrite IHaex2; reflexivity.
    + destruct (constant_fold aex1); destruct (constant_fold aex2); simpl in *;
      rewrite IHaex1; rewrite IHaex2; reflexivity.
Qed.

Theorem algebraic_simpl_ok :
    forall aex, aeval (algebraic_simpl aex) = aeval aex.
Proof.
    induction aex; simpl.
    + reflexivity.
    + destruct (algebraic_simpl aex1); try destruct n;
      destruct (algebraic_simpl aex2); try destruct n0; try destruct n; simpl in *;
      try rewrite <- IHaex1; try rewrite <- IHaex2; simpl; try rewrite <- plus_n_O;
      reflexivity.
    + destruct (algebraic_simpl aex1); try destruct n;
      destruct (algebraic_simpl aex2); try destruct n0; try destruct n; simpl in *;
      try rewrite <- IHaex1; try rewrite <- IHaex2; simpl;
      try rewrite <- mult_n_O; try rewrite <- plus_n_O; try reflexivity;
      try destruct n0; try destruct n; simpl; try rewrite Nat.mul_1_r; try reflexivity.
Qed.

Theorem comp_ok :
    forall f1 f2,
    (forall aex, aeval (f1 aex) = aeval aex) ->
    (forall aex, aeval (f2 aex) = aeval aex) ->
    forall aex, aeval (f1 (f2 aex)) = aeval aex.
Proof.
    intros.
    rewrite H. rewrite H0. reflexivity.
Qed.

Theorem specific_comp_ok :
    forall aex, aeval (constant_fold (algebraic_simpl aex)) = aeval aex.
Proof.
    apply (comp_ok constant_fold algebraic_simpl constant_fold_ok algebraic_simpl_ok).
Qed.
