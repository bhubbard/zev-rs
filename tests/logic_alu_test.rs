use zev::logic::{
    AluMode, AluOp, GateType, SemanticAluEngine, SemanticStateAlu, TopologicalCircuit,
    ZevLogicEngine,
};

#[test]
fn test_logic_engine_gates() {
    let mut engine = ZevLogicEngine::new();

    // NAND
    assert_eq!(engine.nand(false, false), (true, 1.0));
    assert_eq!(engine.nand(true, true), (false, 1.0));

    // AND
    assert_eq!(engine.and(true, true), (true, 1.0));
    assert_eq!(engine.and(true, false), (false, 1.0));

    // OR
    assert_eq!(engine.or(false, false), (false, 1.0));
    assert_eq!(engine.or(true, false), (true, 1.0));

    // XOR
    assert_eq!(engine.xor(true, true), (false, 1.0));
    assert_eq!(engine.xor(true, false), (true, 1.0));

    // MUX
    assert_eq!(engine.mux(false, true, false), (true, 1.0)); // sel=false selects a (true)
    assert_eq!(engine.mux(true, true, false), (false, 1.0)); // sel=true selects b (false)
}

#[test]
fn test_probabilistic_gate_and_abstention() {
    let mut engine = ZevLogicEngine::new();

    // Probabilistic NAND
    let res = engine
        .evaluate_probabilistic(GateType::Nand, &[0.90, 0.90], 0.05)
        .unwrap();
    assert!(!res.decision); // 1 - 0.81 = 0.19 (< 0.5)
    assert!(!res.abstained);

    // Borderline abstention case
    let border = engine
        .evaluate_probabilistic(GateType::Nand, &[0.7071, 0.7071], 0.05)
        .unwrap();
    assert!(border.abstained);
}

#[test]
fn test_topological_dag_concurrency() {
    let mut circuit = TopologicalCircuit::new();
    let a = circuit.alloc_wire();
    let b = circuit.alloc_wire();
    let cin = circuit.alloc_wire();
    circuit.set_inputs(&[a, b, cin]);

    let (sum, cout) = circuit.add_nand_full_adder(a, b, cin);
    circuit.set_outputs(&[sum, cout]);

    circuit.compile().unwrap();
    assert!(circuit.stages.len() >= 3);

    // 1 + 1 + 0 = 0 (carry 1)
    let rep_seq = circuit.evaluate_sequential(&[true, true, false]).unwrap();
    assert_eq!(rep_seq.outputs, vec![false, true]);

    let rep_par = circuit.evaluate_parallel(&[true, true, false]).unwrap();
    assert_eq!(rep_par.outputs, vec![false, true]);
    assert_eq!(rep_par.compound_confidence, 1.0);
}

#[test]
fn test_alu_7_plus_5_equals_12() {
    let mut alu = SemanticAluEngine::new();

    // 1. Structural NAND (circuit of gates)
    let struct_rep = alu
        .execute(AluOp::Add, 7, 5, 4, AluMode::StructuralNand)
        .unwrap();
    assert_eq!(struct_rep.result, 12);
    assert!(!struct_rep.carry_out);
    assert!(!struct_rep.zero_flag);
    assert_eq!(struct_rep.compound_confidence, 1.0);
    assert_eq!(struct_rep.cost_dollars, 0.0);
    assert!(struct_rep.gate_evaluations > 30);

    // 2. Macro-Gate Semantic ALU (1 decision step)
    let macro_rep = alu
        .execute(AluOp::Add, 7, 5, 4, AluMode::MacroSemantic)
        .unwrap();
    assert_eq!(macro_rep.result, 12);
    assert_eq!(macro_rep.gate_evaluations, 1);
    assert_eq!(macro_rep.critical_path_depth, 1);
    assert_eq!(macro_rep.compound_confidence, 1.0);
}

#[test]
fn test_alu_subtraction_and_flags() {
    let mut alu = SemanticAluEngine::new();

    // 12 - 5 = 7
    let r1 = alu
        .execute(AluOp::Sub, 12, 5, 4, AluMode::MacroSemantic)
        .unwrap();
    assert_eq!(r1.result, 7);
    assert!(!r1.zero_flag);

    // 5 - 5 = 0 (Zero flag = true)
    let r2 = alu
        .execute(AluOp::Sub, 5, 5, 4, AluMode::MacroSemantic)
        .unwrap();
    assert_eq!(r2.result, 0);
    assert!(r2.zero_flag);
}

#[test]
fn test_game_semantic_state_transitions() {
    // Jacob Composure + Opponent Aggression
    let (prob, dec, conf) = SemanticStateAlu::combine_states(0.85, 0.70, GateType::Or);
    assert!(dec);
    assert!(prob > 0.90);
    assert!(conf > 0.80);
}
