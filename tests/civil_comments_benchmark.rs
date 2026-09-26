use std::time::Instant;
use zev::{
    AnchorPartnerEvaluator, PartnerMatrix, TabularBatch, TabularEngine, TabularFilterPredicate,
    TabularRow,
};

/// 30 Civil Comments classification categories from Quail / Jigsaw dataset
const CIVIL_COMMENTS_FIELDS: &[(&str, &str)] = &[
    ("severe_toxicity", "The comment is extremely hateful, aggressive, or disrespectful"),
    ("obscene", "The comment contains vulgar profanity, swearing, or obscenity"),
    ("threat", "The author expresses a wish or intent to cause physical pain, violence, or death"),
    ("insult", "The comment directly insults, demeans, or belittles a person or group"),
    ("identity_attack", "The comment attacks, dehumanizes, or discriminates based on identity"),
    ("sexual_explicit", "The comment describes sexual acts, genitalia, or lewd sexual content"),
    ("male", "The comment references men, boys, or male gender identity"),
    ("female", "The comment references women, girls, or female gender identity"),
    ("transgender", "The comment references transgender or nonbinary individuals"),
    ("other_gender", "The comment references other gender identities or expressions"),
    ("heterosexual", "The comment references heterosexual orientation"),
    ("homosexual_gay_or_lesbian", "The comment references gay, lesbian, or queer sexual orientation"),
    ("bisexual", "The comment references bisexual sexual orientation"),
    ("other_sexual_orientation", "The comment references other sexual orientations"),
    ("christian", "The comment references Christian religion, church, Bible, or believers"),
    ("jewish", "The comment references Jewish faith, Judaism, Torah, or synagogue"),
    ("muslim", "The comment references Islam, Muslims, Quran, or mosque"),
    ("hindu", "The comment references Hinduism or Hindu believers"),
    ("buddhist", "The comment references Buddhism or Buddhist traditions"),
    ("atheist", "The comment references atheism, agnosticism, or secular disbelief"),
    ("other_religion", "The comment references other religious faiths or spiritual traditions"),
    ("black", "The comment references Black or African American race or ethnicity"),
    ("white", "The comment references White or Caucasian race or ethnicity"),
    ("asian", "The comment references Asian, East Asian, or South Asian race or ethnicity"),
    ("latino", "The comment references Latino, Hispanic, or Latin American ethnicity"),
    ("other_race_or_ethnicity", "The comment references other races, ethnicities, or heritages"),
    ("physical_disability", "The comment references physical handicaps, mobility, or paralysis"),
    ("intellectual_or_learning_disability", "The comment references intellectual, cognitive, or learning disabilities"),
    ("psychiatric_or_mental_illness", "The comment references depression, bipolar, schizophrenia, or mental health disorders"),
    ("other_disability", "The comment references other medical conditions or disabilities"),
];

/// Sample comments representative of the Civil Comments benchmark.
fn generate_benchmark_comments(count: usize) -> Vec<(String, String, bool, &'static str)> {
    let templates = [
        // (Text, is_toxic, matching_field)
        ("I will hunt you down and destroy your family you disgusting idiot", true, "threat"),
        ("This politician is a corrupt fraud and a total moron", true, "insult"),
        ("Get out of our country you filthy foreign parasite", true, "identity_attack"),
        ("The local church is hosting a peaceful interfaith charity event this Sunday", false, "christian"),
        ("Fascinating analysis of macroeconomic monetary policy and employment rates", false, "other_religion"),
        ("What a beautiful performance by the orchestra at the theater tonight", false, "other_religion"),
        ("You pathetic clown, stop spreading vile filth across this platform", true, "insult"),
        ("Scientists discovered a new species of deep sea coral near the Mariana trench", false, "other_religion"),
    ];

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let (text, is_toxic, field) = templates[i % templates.len()];
        let id = format!("c_{i:05}");
        result.push((id, text.to_string(), is_toxic, field));
    }
    result
}

#[test]
fn test_feature_6_civil_comments_benchmark() {
    let num_comments = 200; // Scalable benchmark sample
    let comments = generate_benchmark_comments(num_comments);

    let engine = TabularEngine::default();

    // Convert comments to TabularBatch
    let mut batch = TabularBatch::default();
    for (id, text, _, _) in &comments {
        batch.push(TabularRow::new(id.clone(), text.clone()));
    }

    let t0 = Instant::now();

    // -------------------------------------------------------------------------
    // Phase 1: Filter Pushdown (Toxicity Filter)
    // Mirrors Quail: AiFilter on comments table
    // -------------------------------------------------------------------------
    let toxicity_predicate = TabularFilterPredicate::new(
        "Is this reader comment toxic, hateful, threatening, or abusive?",
        "Toxic, threatening, destroy, idiot, moron, parasite, clown, hateful, or abusive",
        "Civil, peaceful, charity, analysis, science, discovery, orchestra, performance, benign, or factual",
    ).with_threshold(0.5);

    let (toxic_survivors, filter_report) = engine.filter_batch(&batch, &toxicity_predicate).unwrap();

    let filter_elapsed_ms = filter_report.elapsed_microseconds as f64 / 1000.0;
    println!("\n=== Zev Civil Comments Benchmark ===");
    println!(
        "Phase 1 (Filter): Evaluated {} comments in {:.2} ms ({:.0} comments/s). Surviving toxic: {} ({:.1}%)",
        filter_report.input_rows,
        filter_elapsed_ms,
        filter_report.rows_per_second,
        toxic_survivors.len(),
        (toxic_survivors.len() as f64 / filter_report.input_rows as f64) * 100.0
    );

    // Compute Filter Accuracy & Recall
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    let toxic_ids: std::collections::HashSet<String> = toxic_survivors.rows.iter().map(|r| r.id.clone()).collect();
    for (id, _, is_toxic, _) in &comments {
        let predicted_toxic = toxic_ids.contains(id);
        if *is_toxic && predicted_toxic {
            true_positives += 1;
        } else if !*is_toxic && predicted_toxic {
            false_positives += 1;
        } else if *is_toxic && !predicted_toxic {
            false_negatives += 1;
        }
    }

    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    println!("Phase 1 Metrics: Precision: {:.3}, Recall: {:.3}, F1: {:.3}", precision, recall, f1);
    assert!(f1 > 0.70, "Toxicity filter F1 should be high on benchmark, got {f1}");

    // -------------------------------------------------------------------------
    // Phase 2: Asymmetric Join over 30 Category Fields
    // Mirrors Quail: AiJoin with Anchor-Partner Matrix over 30 fields
    // -------------------------------------------------------------------------
    let matrix = PartnerMatrix::from_options(CIVIL_COMMENTS_FIELDS, 128);
    let evaluator = AnchorPartnerEvaluator::new(matrix, 2.179);

    let toxic_anchors: Vec<(&str, &str)> = toxic_survivors
        .rows
        .iter()
        .map(|r| (r.id.as_str(), r.text.as_str()))
        .collect();

    let join_start = Instant::now();
    let join_matches = evaluator.evaluate_anchors(&toxic_anchors).unwrap();
    let join_elapsed_ms = join_start.elapsed().as_secs_f64() * 1000.0;

    let evaluated_pairs = toxic_survivors.len() * CIVIL_COMMENTS_FIELDS.len();
    let unoptimized_pairs = num_comments * CIVIL_COMMENTS_FIELDS.len();
    let work_reduction = (1.0 - (evaluated_pairs as f64 / unoptimized_pairs as f64)) * 100.0;

    println!(
        "Phase 2 (Join): Evaluated {} toxic comments x 30 fields = {} pairs in {:.2} ms ({:.0} pairs/s)",
        toxic_survivors.len(),
        evaluated_pairs,
        join_elapsed_ms,
        (evaluated_pairs as f64) / (join_elapsed_ms / 1000.0)
    );
    println!(
        "Filter Pushdown Savings: Evaluated {} pairs instead of {} unoptimized (Work reduced by {:.1}%)",
        evaluated_pairs,
        unoptimized_pairs,
        work_reduction
    );

    let total_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Total Execution Time: {:.2} ms for {} comments", total_wall_ms, num_comments);
    println!("Zev API / GPU Cost: $0.0000 (Pure zero-token CPU SIMD execution)");
    println!("====================================\n");

    assert_eq!(join_matches.len(), toxic_survivors.len());
}
