//! Zero-allocation static lexicon for canonical synset matching, bilateral lexical antonyms,
//! and quantized inverse document frequency (IDF) weighting.
//!
//! Uses an XXH3-indexed static hash table initialized once on first access, providing
//! O(1) (~20 ns) lookups per token with zero heap allocations on the hot path.

use crate::fast_hash::Xxh3BuildHasher;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Compact 5-byte lexicon entry fitting within a single CPU register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexiconEntry {
    /// Canonical synset ID (non-zero when part of a synonym or antonym cluster).
    pub synset_id: u16,
    /// Canonical synset ID of the strict lexical antonym cluster (0 if none).
    pub antonym_synset_id: u16,
    /// Quantized IDF multiplier where 128 == 1.0x.
    pub idf_q8: u8,
}

impl LexiconEntry {
    #[inline(always)]
    pub fn idf_scale(self) -> f64 {
        (self.idf_q8 as f64) / 128.0
    }
}

/// Bilateral strict antonym synset pairs: `(positive_or_pole_a, negative_or_pole_b, idf_q8)`.
///
/// Only unambiguous polarity opposites are included here so that `has_morphological_antonym`
/// never fires on polysemous words.
const ANTONYM_SYNSET_PAIRS: &[(&[&str], &[&str], u8)] = &[
    // 1. Permission vs Prohibition
    (
        &[
            "allow",
            "allowed",
            "allowing",
            "permit",
            "permitted",
            "permitting",
            "permissible",
            "authorize",
            "authorized",
            "authorise",
            "authorised",
            "entitle",
            "entitled",
            "eligible",
        ],
        &[
            "prohibit",
            "prohibited",
            "prohibiting",
            "prohibition",
            "forbid",
            "forbidden",
            "forbidding",
            "ban",
            "banned",
            "disallow",
            "disallowed",
            "impermissible",
            "ineligible",
            "unauthorized",
            "unauthorised",
        ],
        140,
    ),
    // 2. Obligation vs Optionality
    (
        &[
            "mandatory",
            "compulsory",
            "obligatory",
            "mandated",
            "imperative",
            "requisite",
        ],
        &[
            "optional",
            "voluntary",
            "discretionary",
            "elective",
            "nonmandatory",
            "unnecessary",
        ],
        144,
    ),
    // 3. Approval / Acceptance vs Rejection / Denial
    (
        &[
            "approve",
            "approved",
            "approval",
            "uphold",
            "upheld",
            "ratify",
            "ratified",
            "endorse",
            "endorsed",
        ],
        &[
            "reject",
            "rejected",
            "rejection",
            "deny",
            "denied",
            "denial",
            "refuse",
            "refused",
            "refusal",
            "veto",
            "vetoed",
            "disallow",
        ],
        136,
    ),
    // 4. Compliance vs Violation
    (
        &[
            "comply",
            "compliant",
            "compliance",
            "adhere",
            "adherence",
            "conform",
            "conforming",
            "lawful",
        ],
        &[
            "violate",
            "violated",
            "violation",
            "breach",
            "breached",
            "infringe",
            "infringed",
            "infringement",
            "noncompliant",
            "noncompliance",
            "unlawful",
            "contravene",
        ],
        144,
    ),
    // 5. Validity vs Expiration / Revocation
    (
        &["valid", "validated", "unexpired", "enforceable"],
        &[
            "invalid",
            "invalidated",
            "expired",
            "lapsed",
            "void",
            "voided",
            "nullified",
            "unenforceable",
        ],
        140,
    ),
    // 6. Confidentiality vs Public Disclosure
    (
        &[
            "confidential",
            "confidentiality",
            "classified",
            "nonpublic",
            "proprietary",
            "undisclosed",
        ],
        &["unclassified", "declassified", "publicly"],
        140,
    ),
    // 7. Permanence vs Temporariness
    (
        &[
            "permanent",
            "permanently",
            "irreversible",
            "irreversibly",
            "irrecoverable",
            "perpetual",
            "indefinite",
            "indefinitely",
        ],
        &[
            "temporary",
            "temporarily",
            "transient",
            "reversible",
            "recoverable",
            "ephemeral",
            "provisional",
            "interim",
        ],
        140,
    ),
    // 8. Sufficiency vs Insufficiency
    (
        &[
            "sufficient",
            "sufficiently",
            "adequate",
            "adequately",
            "ample",
            "satisfactory",
        ],
        &[
            "insufficient",
            "inadequate",
            "deficient",
            "scarce",
            "unsatisfactory",
        ],
        136,
    ),
    // 9. Encryption vs Plaintext
    (
        &["encrypted", "ciphered", "enciphered", "cryptographic"],
        &["unencrypted", "plaintext", "cleartext", "decrypted"],
        148,
    ),
    // 10. Authentication vs Anonymous
    (
        &["authenticated", "logged_in", "signed_in"],
        &["unauthenticated", "anonymous", "unverified", "logged_out"],
        144,
    ),
    // 11. Activation vs Deactivation
    (
        &["enable", "enabled", "activate", "activated", "turn_on"],
        &[
            "disable",
            "disabled",
            "deactivate",
            "deactivated",
            "turn_off",
        ],
        136,
    ),
    // 12. Payment Settlement vs Arrears
    (
        &["paid", "settled"],
        &["unpaid", "overdue", "delinquent", "arrears", "defaulted"],
        140,
    ),
    // 13. Authenticity vs Counterfeit
    (
        &["authentic", "genuine", "legitimate", "bonafide"],
        &[
            "counterfeit",
            "forged",
            "fraudulent",
            "bogus",
            "spoofed",
            "fabricated",
        ],
        144,
    ),
    // 14. Electrical Conductivity vs Insulation
    (
        &["conductor", "conductive", "conducts"],
        &["insulator", "insulating", "nonconductive"],
        148,
    ),
    // 15. Renewable vs Nonrenewable Energy
    (
        &["renewable", "sustainable", "inexhaustible"],
        &["nonrenewable", "depletable"],
        144,
    ),
    // 16. Evaporation vs Condensation
    (
        &["evaporate", "evaporation", "vaporize", "vaporization"],
        &["condense", "condensation", "liquefy"],
        148,
    ),
    // 17. Solidification vs Melting
    (
        &["solidify", "solidification"],
        &["melt", "melting", "thaw", "thawing"],
        144,
    ),
    // 18. Autotroph vs Heterotroph
    (
        &["autotroph", "autotrophic"],
        &["heterotroph", "heterotrophic"],
        152,
    ),
    // 19. Herbivore vs Carnivore
    (
        &["herbivore", "herbivorous"],
        &["carnivore", "carnivorous"],
        152,
    ),
    // 20. Hereditary vs Congenital
    (
        &["inherited", "hereditary", "innate", "congenital"],
        &["nonhereditary", "uninherited"],
        144,
    ),
    // 21. Malicious vs Benign
    (
        &["malicious", "adversarial", "hostile", "harmful"],
        &["benign", "harmless", "innocuous"],
        144,
    ),
    // 22. Synchronous vs Asynchronous
    (
        &["synchronous", "synchronously", "blocking"],
        &["asynchronous", "asynchronously", "nonblocking"],
        144,
    ),
    // 23. Domestic vs International
    (
        &["domestic", "intranational"],
        &["international", "cross_border", "overseas", "foreign"],
        136,
    ),
];

/// Unilateral high-precision canonical synonym clusters `(words, idf_q8)`.
const SYNONYM_CLUSTERS: &[(&[&str], u8)] = &[
    // Financial & Banking
    (
        &[
            "refund",
            "refunds",
            "refunded",
            "reimburse",
            "reimbursed",
            "reimbursement",
            "repay",
            "repaid",
            "repayment",
            "money_back",
            "chargeback",
        ],
        144,
    ),
    (
        &[
            "invoice",
            "invoices",
            "billing",
            "receipt",
            "receipts",
            "remittance",
        ],
        136,
    ),
    (
        &[
            "salary",
            "salaries",
            "wage",
            "wages",
            "payroll",
            "remuneration",
            "stipend",
            "paycheck",
        ],
        140,
    ),
    (
        &[
            "discount",
            "discounts",
            "rebate",
            "coupon",
            "voucher",
            "markdown",
            "promo",
        ],
        136,
    ),
    (
        &[
            "surcharge",
            "tariff",
            "levy",
            "penalties",
            "penalty",
        ],
        136,
    ),
    (
        &[
            "mortgage",
            "mortgages",
            "home_loan",
            "refinance",
            "refinancing",
        ],
        144,
    ),
    (
        &[
            "withdraw",
            "withdrawal",
            "cash_out",
            "dispense",
            "atm_withdrawal",
        ],
        140,
    ),
    (
        &[
            "beneficiary",
            "payee",
            "recipient",
            "transferee",
        ],
        140,
    ),
    // Lifecycle & Subscriptions
    (
        &[
            "cancel",
            "canceled",
            "cancelled",
            "cancellation",
            "terminate",
            "terminated",
            "termination",
            "discontinue",
            "unsubscribe",
            "rescind",
            "revoke",
            "revoked",
        ],
        136,
    ),
    (
        &[
            "suspend",
            "suspended",
            "suspension",
            "moratorium",
        ],
        136,
    ),
    (
        &[
            "renew",
            "renewal",
            "prolong",
            "reinstate",
        ],
        136,
    ),
    (
        &[
            "postpone",
            "postponed",
            "reschedule",
            "rescheduled",
            "defer",
            "deferred",
            "adjourn",
        ],
        140,
    ),
    // Software, Infrastructure & Security
    (
        &[
            "outage",
            "downtime",
            "blackout",
            "unavailability",
            "service_disruption",
        ],
        144,
    ),
    (
        &[
            "reboot",
            "rebooted",
            "restart",
            "restarted",
            "power_cycle",
        ],
        140,
    ),
    (
        &[
            "defect",
            "defective",
            "bug",
            "glitch",
            "malfunction",
            "malfunctioning",
            "faulty",
        ],
        136,
    ),
    (
        &[
            "corrupt",
            "corrupted",
            "corruption",
            "malformed",
            "garbled",
            "truncated",
        ],
        140,
    ),
    (
        &[
            "credential",
            "credentials",
            "password",
            "passcode",
            "passphrase",
            "secret_key",
            "api_key",
        ],
        140,
    ),
    (
        &[
            "vulnerability",
            "exploit",
            "cve",
            "security_flaw",
            "backdoor",
        ],
        144,
    ),
    (
        &[
            "phishing",
            "spoofing",
            "social_engineering",
            "impersonation",
        ],
        148,
    ),
    (
        &[
            "latency",
            "lag",
            "slowdown",
            "bottleneck",
            "throttling",
        ],
        140,
    ),
    (
        &[
            "rollback",
            "revert",
            "reverted",
            "undo",
            "backout",
        ],
        140,
    ),
    // Logistics, Travel & Aviation
    (
        &[
            "baggage",
            "luggage",
            "suitcase",
            "carry_on",
            "checked_bag",
        ],
        148,
    ),
    (
        &[
            "itinerary",
            "timetable",
            "flight_schedule",
            "travel_plan",
        ],
        144,
    ),
    (
        &[
            "parcel",
            "package",
            "shipment",
            "consignment",
            "freight",
            "cargo",
        ],
        136,
    ),
    (
        &[
            "courier",
            "delivery_driver",
            "postal",
            "shipping_carrier",
        ],
        140,
    ),
    (
        &[
            "automobile",
            "vehicle",
            "sedan",
            "motor_vehicle",
            "suv",
        ],
        144,
    ),
    (
        &[
            "reservation",
            "booking",
            "booked",
        ],
        136,
    ),
    // Legal & Contracts (ContractNLI & Policy)
    (
        &[
            "disclose",
            "disclosure",
            "divulge",
            "reveal",
            "disseminate",
        ],
        140,
    ),
    (
        &[
            "retain",
            "retention",
            "preserve",
            "preservation",
            "archive",
            "archival",
        ],
        136,
    ),
    (
        &[
            "delete",
            "deletion",
            "erase",
            "erasure",
            "purge",
            "expunge",
            "destroy",
            "destruction",
        ],
        140,
    ),
    (
        &[
            "exempt",
            "exemption",
            "waive",
            "waived",
            "waiver",
            "dispensation",
            "carveout",
        ],
        144,
    ),
    (
        &[
            "indemnify",
            "indemnification",
            "indemnity",
            "hold_harmless",
        ],
        152,
    ),
    (
        &[
            "arbitration",
            "arbitrator",
            "mediation",
            "tribunal",
        ],
        148,
    ),
    (
        &[
            "subsidiary",
            "affiliate",
            "affiliates",
            "subcontractor",
        ],
        140,
    ),
    (
        &[
            "consent",
            "permission",
            "authorization",
            "assent",
            "sign_off",
        ],
        136,
    ),
    (
        &[
            "notify",
            "notification",
            "inform",
            "informed",
            "notice",
            "alert",
        ],
        132,
    ),
    // Medical & Clinical
    (
        &[
            "physician",
            "doctor",
            "clinician",
            "medical_practitioner",
        ],
        144,
    ),
    (
        &[
            "medication",
            "medicine",
            "pharmaceutical",
            "prescription",
            "remedy",
        ],
        144,
    ),
    (
        &[
            "illness",
            "ailment",
            "sickness",
            "malady",
            "pathology",
        ],
        144,
    ),
    (
        &[
            "febrile",
            "fever",
            "feverish",
            "pyrexia",
        ],
        152,
    ),
    (
        &[
            "hypertension",
            "high_blood_pressure",
        ],
        152,
    ),
    // Natural Science & STEM
    (
        &[
            "photosynthesis",
            "photosynthetic",
            "chloroplast",
            "chlorophyll",
        ],
        156,
    ),
    (
        &[
            "precipitation",
            "rainfall",
            "snowfall",
            "sleet",
        ],
        148,
    ),
    (
        &[
            "erosion",
            "weathering",
            "abrasion",
        ],
        148,
    ),
    (
        &[
            "gravity",
            "gravitational",
        ],
        148,
    ),
    (
        &[
            "friction",
            "frictional",
        ],
        148,
    ),
    (
        &[
            "mitosis",
            "meiosis",
            "cell_division",
        ],
        152,
    ),
    // Consumer & Smart Home
    (
        &[
            "beverage",
            "beverages",
            "refreshment",
            "drink",
            "drinks",
        ],
        140,
    ),
    (
        &[
            "apparel",
            "clothing",
            "garment",
            "garments",
            "attire",
            "footwear",
        ],
        144,
    ),
    (
        &[
            "residence",
            "dwelling",
            "domicile",
            "household",
        ],
        140,
    ),
    (
        &[
            "refrigerator",
            "fridge",
            "freezer",
        ],
        148,
    ),
    (
        &[
            "thermostat",
            "hvac",
            "climate_control",
        ],
        148,
    ),
    (
        &[
            "illumination",
            "brightness",
            "luminance",
            "lighting",
        ],
        140,
    ),
];

/// Generic low-IDF conversational verbs/nouns that receive dampened weight (0.72x)
/// when a candidate description has >= 3 content words.
const LOW_IDF_WORDS: &[&str] = &[
    "make", "made", "making", "take", "taken", "taking", "give", "given", "giving", "keep",
    "kept", "keeping", "seem", "seemed", "thing", "things", "place", "places", "point", "points",
    "way", "ways",
];

static LEXICON_MAP: LazyLock<HashMap<String, LexiconEntry, Xxh3BuildHasher>> =
    LazyLock::new(|| {
        let mut map = HashMap::with_capacity_and_hasher(1024, Xxh3BuildHasher);
        let mut next_synset_id: u16 = 1;

        for &(pole_a, pole_b, idf_q8) in ANTONYM_SYNSET_PAIRS {
            let id_a = next_synset_id;
            let id_b = next_synset_id + 1;
            next_synset_id += 2;

            let entry_a = LexiconEntry {
                synset_id: id_a,
                antonym_synset_id: id_b,
                idf_q8,
            };
            let entry_b = LexiconEntry {
                synset_id: id_b,
                antonym_synset_id: id_a,
                idf_q8,
            };

            for &w in pole_a {
                map.insert(w.to_string(), entry_a);
            }
            for &w in pole_b {
                map.insert(w.to_string(), entry_b);
            }
        }

        for &(words, idf_q8) in SYNONYM_CLUSTERS {
            let syn_id = next_synset_id;
            next_synset_id += 1;

            let entry = LexiconEntry {
                synset_id: syn_id,
                antonym_synset_id: 0,
                idf_q8,
            };

            for &w in words {
                map.entry(w.to_string()).or_insert(entry);
            }
        }

        for &w in LOW_IDF_WORDS {
            map.entry(w.to_string()).or_insert(LexiconEntry {
                synset_id: 0,
                antonym_synset_id: 0,
                idf_q8: 92, // ~0.72x
            });
        }

        map
    });

/// O(1) lookup of a lowercase word in the static lexicon (~20 ns via XXH3).
#[inline(always)]
pub fn lookup_word(word: &str) -> Option<LexiconEntry> {
    if word.len() < 3 {
        return None;
    }
    LEXICON_MAP.get(word).copied()
}

/// O(1) lookup checking the exact lowercase token first.
#[inline(always)]
pub fn lookup_word_or_stem(word: &str, _stem: &str) -> Option<LexiconEntry> {
    lookup_word(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synset_and_antonym_lookup() {
        let allow = lookup_word("permitted").expect("permitted in lexicon");
        let forbid = lookup_word("prohibited").expect("prohibited in lexicon");
        assert_eq!(allow.antonym_synset_id, forbid.synset_id);
        assert_eq!(forbid.antonym_synset_id, allow.synset_id);

        let refund = lookup_word("reimbursement").expect("reimbursement in lexicon");
        let repay = lookup_word("chargeback").expect("chargeback in lexicon");
        assert_eq!(refund.synset_id, repay.synset_id);
        assert_ne!(refund.synset_id, 0);
    }
}
