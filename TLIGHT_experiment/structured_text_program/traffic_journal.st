PROGRAM program0
  VAR
    p_red AT %QX100.0 : BOOL;
    p_green AT %QX100.1 : BOOL;
    v_red AT %QX100.2 : BOOL;
    v_yel AT %QX100.3 : BOOL;
    v_green AT %QX100.4 : BOOL;
    p_right AT %QX0.0 : BOOL;
    p_left AT %QX0.1 : BOOL;
    start AT %QX0.2 : BOOL;
  END_VAR
  VAR
    T2 : BOOL;
    T3 : BOOL;
    T4 : BOOL;
    T5 : BOOL;
    T6 : BOOL;
    TON0 : TON;
    TON1 : TON;
    TON2 : TON;
    TON3 : TON;
  END_VAR
  VAR
    C1 AT %MW1 : INT;
    C2 AT %MW2 : INT;
    C3 AT %MW3 : INT;
    C4 AT %MW4 : INT;
    C5 AT %MW5 : INT;
  END_VAR
  VAR
    TON4 : TON;
    TON5 : TON;
    TIME_TO_INT61_OUT : INT;
    TIME_TO_INT42_OUT : INT;
    TIME_TO_INT63_OUT : INT;
    TIME_TO_INT35_OUT : INT;
  END_VAR

  start := NOT(p_right) AND (NOT(p_left) OR start);
  v_green := NOT(v_red) AND NOT(v_yel) AND start;
  TON1(IN := NOT(T4) AND start, PT := T#60s);
  v_yel := NOT(v_red) AND TON1.Q;
  TON0(IN := NOT(T4) AND start, PT := T#64s);
  v_red := TON0.Q;
  p_red := (NOT(v_red) OR T2) AND start;
  p_green := NOT(p_red) AND start;
  TON2(IN := p_green, PT := T#10s);
  T2 := NOT(T4) AND T2 OR TON2.Q;
  TIME_TO_INT61_OUT := TIME_TO_INT(TON2.ET);
  C3 := TIME_TO_INT61_OUT;
  TON4(IN := v_yel, PT := T#3s);
  TIME_TO_INT42_OUT := TIME_TO_INT(TON4.ET);
  C1 := TIME_TO_INT42_OUT;
  C2 := TIME_TO_INT42_OUT;
  TON3(IN := T2, PT := T#6s);
  T3 := NOT(T4) AND v_red AND T3 OR TON3.Q;
  TIME_TO_INT63_OUT := TIME_TO_INT(TON3.ET);
  C4 := TIME_TO_INT63_OUT;
  TON5(IN := T3, PT := T#2s);
  T4 := TON5.Q;
  TIME_TO_INT35_OUT := TIME_TO_INT(TON5.ET);
  C5 := TIME_TO_INT35_OUT;
END_PROGRAM


CONFIGURATION Config0

  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#20ms,PRIORITY := 0);
    PROGRAM instance0 WITH task0 : program0;
  END_RESOURCE
END_CONFIGURATION
