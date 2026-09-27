//! Hierarchical intent classifier and domain sieve for high-cardinality intent tasks
//! (BANKING77, CLINC150, customer support).

/// Detect domain and apply logit adjustments to candidate options
pub fn apply_hierarchical_intent_sieve(state: &str, logits: &mut [f64], candidate_ids: &[&str]) {
    let state_lower = state.to_lowercase();

    // Banking domains
    let is_card = state_lower.contains("card")
        || state_lower.contains("debit")
        || state_lower.contains("credit")
        || state_lower.contains("pin")
        || state_lower.contains("cvv")
        || state_lower.contains("stolen")
        || state_lower.contains("lost")
        || state_lower.contains("contactless")
        || state_lower.contains("plastic");

    let is_transfer = state_lower.contains("transfer")
        || state_lower.contains("wire")
        || state_lower.contains("send money")
        || state_lower.contains("beneficiary")
        || state_lower.contains("iban")
        || state_lower.contains("swift");

    let is_charge_fee = state_lower.contains("charge")
        || state_lower.contains("fee")
        || state_lower.contains("charged twice")
        || state_lower.contains("extra cost")
        || state_lower.contains("penalty")
        || state_lower.contains("interest");

    let is_balance = state_lower.contains("balance")
        || state_lower.contains("how much")
        || state_lower.contains("statement")
        || state_lower.contains("funds available");

    let is_exchange = state_lower.contains("exchange")
        || state_lower.contains("currency")
        || state_lower.contains("rate")
        || state_lower.contains("convert")
        || state_lower.contains("forex");

    // General domains (CLINC150)
    let is_weather = state_lower.contains("weather")
        || state_lower.contains("forecast")
        || state_lower.contains("temperature")
        || state_lower.contains("rain")
        || state_lower.contains("snow")
        || state_lower.contains("sunny");

    let is_alarm = state_lower.contains("alarm")
        || state_lower.contains("timer")
        || state_lower.contains("wake me up")
        || state_lower.contains("remind me");

    let is_travel = state_lower.contains("flight")
        || state_lower.contains("hotel")
        || state_lower.contains("airline")
        || state_lower.contains("car rental")
        || state_lower.contains("reservation");

    let has_token = |id_str: &str, tok: &str| {
        id_str
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| w == tok)
    };

    for (idx, &id) in candidate_ids.iter().enumerate() {
        let id_lower = id.to_lowercase();
        if is_card && (id_lower.contains("card") || has_token(&id_lower, "pin")) {
            logits[idx] += 5.0;
        }
        if is_transfer && (id_lower.contains("transfer") || id_lower.contains("beneficiary")) {
            logits[idx] += 5.0;
        }
        if is_charge_fee && (has_token(&id_lower, "fee") || id_lower.contains("charge")) {
            logits[idx] += 5.0;
        }
        if is_balance && id_lower.contains("balance") {
            logits[idx] += 5.0;
        }
        if is_exchange && (id_lower.contains("exchange") || id_lower.contains("currency")) {
            logits[idx] += 5.0;
        }
        if is_weather && id_lower.contains("weather") {
            logits[idx] += 6.0;
        }
        if is_alarm && (id_lower.contains("alarm") || has_token(&id_lower, "timer")) {
            logits[idx] += 6.0;
        }
        if is_travel && (id_lower.contains("flight") || id_lower.contains("travel")) {
            logits[idx] += 6.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_sieve_card_domain() {
        let state = "I lost my debit card yesterday and need a replacement.";
        let ids = ["card_lost", "transfer_fee", "weather", "change_pin"];
        let mut logits = [0.0; 4];
        apply_hierarchical_intent_sieve(state, &mut logits, &ids);
        assert!(logits[0] > 0.0); // card_lost gets boost
        assert!(logits[3] > 0.0); // change_pin gets boost
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], 0.0);
    }

    #[test]
    fn test_intent_sieve_weather_domain() {
        let state = "Will it rain tomorrow in Seattle?";
        let ids = ["card_lost", "weather_forecast", "wire_transfer"];
        let mut logits = [0.0; 3];
        apply_hierarchical_intent_sieve(state, &mut logits, &ids);
        assert_eq!(logits[0], 0.0);
        assert!(logits[1] > 0.0); // weather gets boost
        assert_eq!(logits[2], 0.0);
    }
}
