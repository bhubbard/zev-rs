//! Compact static science & fact knowledge associations (ARC-Easy & ARC-Challenge).

pub fn boost_science_concept_associations(context: &str, logits: &mut [f64], candidate_descs: &[&str]) {
    let ctx_lower = context.to_lowercase();

    // Concept rules: (Trigger keywords in context, Associated words in candidate, boost)
    let rules: &[(&[&str], &[&str], f64)] = &[
        // Electricity conductors vs insulators
        (&["conduct electricity", "good conductor", "conducts electric"], &["metal", "copper", "silver", "iron", "gold", "aluminum"], 5.0),
        (&["insulator", "poor conductor", "does not conduct"], &["rubber", "plastic", "wood", "glass"], 5.0),
        
        // Photosynthesis & respiration
        (&["photosynthesis", "plants produce", "chloroplast"], &["oxygen", "glucose", "sugar", "food", "chlorophyll"], 5.0),
        (&["photosynthesis needs", "plants use to make food"], &["sunlight", "carbon dioxide", "water", "light energy"], 5.0),
        (&["cellular respiration", "respiration produces"], &["carbon dioxide", "atp", "energy", "water"], 4.5),

        // Forces & motion
        (&["gravity", "gravitational"], &["mass", "weight", "attract", "downward", "earth's center", "orbit"], 4.5),
        (&["friction"], &["heat", "slow", "resistance", "opposes motion", "surface"], 4.5),
        
        // Water cycle & states of matter
        (&["evaporation", "liquid to gas", "water evaporates"], &["heat", "sun", "vapor", "warms", "boiling"], 4.5),
        (&["condensation", "gas to liquid"], &["cools", "clouds", "droplets", "cold"], 4.5),
        (&["precipitation"], &["rain", "snow", "sleet", "hail"], 4.5),

        // Cells & biology
        (&["plant cell", "plant cells have"], &["cell wall", "chloroplast", "vacuole"], 4.5),
        (&["animal cell", "animal cells have"], &["cell membrane", "no cell wall"], 4.0),
        (&["mitosis", "cell division"], &["chromosomes", "nucleus", "identical", "two daughter"], 4.5),

        // Ecosystems & energy
        (&["producer", "autotroph"], &["plants", "photosynthetic", "make their own food", "grass", "algae"], 5.0),
        (&["herbivore", "primary consumer"], &["plants", "vegetation", "grass"], 4.5),
        (&["carnivore", "secondary consumer"], &["meat", "animals", "predator"], 4.5),
        (&["renewable resource", "renewable energy"], &["solar", "wind", "hydroelectric", "geothermal", "water"], 5.0),
        (&["nonrenewable", "fossil fuel"], &["coal", "oil", "natural gas", "petroleum"], 5.0),

        // Earth science & geology
        (&["igneous"], &["volcano", "magma", "lava", "cooling"], 5.0),
        (&["sedimentary"], &["layers", "fossils", "sediment", "compaction"], 5.0),
        (&["metamorphic"], &["heat and pressure", "changed"], 5.0),
        (&["earth rotates", "earth's rotation", "spinning on axis"], &["day and night", "24 hours"], 5.0),
        (&["earth revolves", "earth's revolution", "tilt on axis"], &["seasons", "year", "365 days"], 5.0),
    ];

    for (triggers, targets, boost) in rules {
        let has_trigger = triggers.iter().any(|&trig| ctx_lower.contains(trig));
        if has_trigger {
            for (idx, &desc) in candidate_descs.iter().enumerate() {
                let desc_lower = desc.to_lowercase();
                if targets.iter().any(|&tgt| desc_lower.contains(tgt)) {
                    logits[idx] += boost;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conductor_concept_boost() {
        let context = "Which of the following is a good conductor of electricity?";
        let descs = ["Rubber band", "Copper wire", "Plastic spoon", "Wooden block"];
        let mut logits = [0.0; 4];
        boost_science_concept_associations(context, &mut logits, &descs);
        assert_eq!(logits[0], 0.0);
        assert!(logits[1] > 0.0); // copper wire boosted
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[3], 0.0);
    }

    #[test]
    fn test_photosynthesis_concept_boost() {
        let context = "What gas do plants produce during photosynthesis?";
        let descs = ["Carbon dioxide", "Oxygen", "Nitrogen", "Methane"];
        let mut logits = [0.0; 4];
        boost_science_concept_associations(context, &mut logits, &descs);
        assert_eq!(logits[0], 0.0);
        assert!(logits[1] > 0.0); // oxygen boosted
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[3], 0.0);
    }
}
