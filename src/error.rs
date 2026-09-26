use thiserror::Error;

#[derive(Error, Debug)]
pub enum ZevError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Invalid policy: {0}")]
    InvalidPolicy(String),

    #[error("Slot limit exceeded: {0}")]
    SlotLimitExceeded(String),

    #[error("Decoding error: {0}")]
    DecodingError(String),

    #[error("Calibration error: {0}")]
    CalibrationError(String),

    #[error("Evaluation error: {0}")]
    Evaluation(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ZevError>;
