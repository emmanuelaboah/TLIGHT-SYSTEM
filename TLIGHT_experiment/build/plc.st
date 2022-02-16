TYPE
  LOGLEVEL : (CRITICAL, WARNING, INFO, DEBUG) := INFO;
END_TYPE

FUNCTION_BLOCK LOGGER
  VAR_INPUT
    TRIG : BOOL;
    MSG : STRING;
    LEVEL : LOGLEVEL := INFO;
  END_VAR
  VAR
    TRIG0 : BOOL;
  END_VAR

  IF TRIG AND NOT TRIG0 THEN
  {{
   LogMessage(GetFbVar(LEVEL),(char*)GetFbVar(MSG, .body),GetFbVar(MSG, .len));
  }}
  END_IF;
  TRIG0:=TRIG;
END_FUNCTION_BLOCK



FUNCTION_BLOCK python_eval
  VAR_INPUT
    TRIG : BOOL;
    CODE : STRING;
  END_VAR
  VAR_OUTPUT
    ACK : BOOL;
    RESULT : STRING;
  END_VAR
  VAR
    STATE : DWORD;
    BUFFER : STRING;
    PREBUFFER : STRING;
    TRIGM1 : BOOL;
    TRIGGED : BOOL;
  END_VAR

  {extern void __PythonEvalFB(int, PYTHON_EVAL*);__PythonEvalFB(0, data__);}
END_FUNCTION_BLOCK

FUNCTION_BLOCK python_poll
  VAR_INPUT
    TRIG : BOOL;
    CODE : STRING;
  END_VAR
  VAR_OUTPUT
    ACK : BOOL;
    RESULT : STRING;
  END_VAR
  VAR
    STATE : DWORD;
    BUFFER : STRING;
    PREBUFFER : STRING;
    TRIGM1 : BOOL;
    TRIGGED : BOOL;
  END_VAR

  {extern void __PythonEvalFB(int, PYTHON_EVAL*);__PythonEvalFB(1,(PYTHON_EVAL*)(void*)data__);}
END_FUNCTION_BLOCK

FUNCTION_BLOCK python_gear
  VAR_INPUT
    N : UINT;
    TRIG : BOOL;
    CODE : STRING;
  END_VAR
  VAR_OUTPUT
    ACK : BOOL;
    RESULT : STRING;
  END_VAR
  VAR
    py_eval : python_eval;
    COUNTER : UINT;
    ADD10_OUT : UINT;
    EQ13_OUT : BOOL;
    SEL15_OUT : UINT;
    AND7_OUT : BOOL;
  END_VAR

  ADD10_OUT := ADD(COUNTER, 1);
  EQ13_OUT := EQ(N, ADD10_OUT);
  SEL15_OUT := SEL(EQ13_OUT, ADD10_OUT, 0);
  COUNTER := SEL15_OUT;
  AND7_OUT := AND(EQ13_OUT, TRIG);
  py_eval(TRIG := AND7_OUT, CODE := CODE);
  ACK := py_eval.ACK;
  RESULT := py_eval.RESULT;
END_FUNCTION_BLOCK


PROGRAM program0
  VAR
    p_red : BOOL;
    p_green : BOOL;
    v_red : BOOL;
    v_yel : BOOL;
    v_green : BOOL;
    p_right : BOOL;
    p_left : BOOL;
    start : BOOL;
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
    C1 : INT;
    C2 : INT;
    C3 : INT;
    C4 : INT;
    C5 : INT;
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
