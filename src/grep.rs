use crate::engine::ZevEngine;
use crate::error::Result;
use crate::types::{BooleanQuestion, Policy, Question, ZevRequest};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Configuration options for semantic codebase search
#[derive(Debug, Clone)]
pub struct GrepOptions {
    pub query: String,
    pub path: PathBuf,
    pub limit: usize,
    pub threshold: f64,
    pub json: bool,
    pub include_evidence: bool,
    pub max_files: usize,
}

impl Default for GrepOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            path: PathBuf::from("."),
            limit: 10,
            threshold: 0.3,
            json: false,
            include_evidence: true,
            max_files: 1000,
        }
    }
}

/// Category of extracted code declaration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeclarationKind {
    Function,
    Struct,
    Enum,
    Trait,
    Class,
    Interface,
    TypeAlias,
    Module,
    Constant,
    Macro,
    Heading,
    Route,
    Other,
}

impl std::fmt::Display for DeclarationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Function => write!(f, "Function"),
            Self::Struct => write!(f, "Struct"),
            Self::Enum => write!(f, "Enum"),
            Self::Trait => write!(f, "Trait"),
            Self::Class => write!(f, "Class"),
            Self::Interface => write!(f, "Interface"),
            Self::TypeAlias => write!(f, "Type"),
            Self::Module => write!(f, "Module"),
            Self::Constant => write!(f, "Const"),
            Self::Macro => write!(f, "Macro"),
            Self::Heading => write!(f, "Heading"),
            Self::Route => write!(f, "Route"),
            Self::Other => write!(f, "Decl"),
        }
    }
}

/// An extracted declaration from source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDeclaration {
    pub file: String,
    pub name: String,
    pub kind: DeclarationKind,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: String,
    pub snippet: String,
}

/// A matched declaration with semantic relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepMatch {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub name: String,
    pub kind: String,
    pub signature: String,
    pub score: f64,
    pub snippet: String,
}

/// Full search result report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepReport {
    pub query: String,
    pub path: String,
    pub files_scanned: usize,
    pub declarations_evaluated: usize,
    pub matches: Vec<GrepMatch>,
    pub duration_ms: f64,
}

/// Detects whether a file path has an extension supported for semantic search
pub fn is_supported_source_file(path: &Path) -> bool {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_lowercase(),
        None => return false,
    };

    matches!(
        ext.as_str(),
        "rs" | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "mjs"
            | "cjs"
            | "py"
            | "go"
            | "c"
            | "cpp"
            | "cc"
            | "h"
            | "hpp"
            | "java"
            | "kt"
            | "swift"
            | "rb"
            | "php"
            | "cs"
            | "sh"
            | "toml"
            | "yaml"
            | "yml"
            | "json"
            | "sql"
            | "md"
    )
}

/// Extracts declarations from file content based on file extension
pub fn extract_declarations(file_path: &str, content: &str) -> Vec<CodeDeclaration> {
    let path = Path::new(file_path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let lines: Vec<&str> = content.lines().collect();

    match ext.as_str() {
        "rs" => extract_rust_declarations(file_path, &lines),
        "py" => extract_python_declarations(file_path, &lines),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => {
            extract_ts_js_declarations(file_path, &lines)
        }
        "go" => extract_go_declarations(file_path, &lines),
        "md" => extract_markdown_headings(file_path, &lines),
        _ => extract_generic_declarations(file_path, &lines),
    }
}

fn build_snippet(lines: &[&str], start_idx: usize, max_lines: usize) -> (String, usize) {
    let end_idx = (start_idx + max_lines).min(lines.len());
    let snippet = lines[start_idx..end_idx].join("\n");
    (snippet, end_idx)
}

fn extract_rust_declarations(file_path: &str, lines: &[&str]) -> Vec<CodeDeclaration> {
    let mut decls = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.is_empty() {
            continue;
        }

        let (kind, name) = if trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ") {
            let rest = trimmed
                .strip_prefix("pub fn ")
                .unwrap_or_else(|| trimmed.strip_prefix("fn ").unwrap_or(""));
            let fn_name = rest.split('(').next().unwrap_or("").trim();
            (DeclarationKind::Function, format!("fn {fn_name}"))
        } else if trimmed.starts_with("pub async fn ") || trimmed.starts_with("async fn ") {
            let rest = trimmed
                .strip_prefix("pub async fn ")
                .unwrap_or_else(|| trimmed.strip_prefix("async fn ").unwrap_or(""));
            let fn_name = rest.split('(').next().unwrap_or("").trim();
            (DeclarationKind::Function, format!("async fn {fn_name}"))
        } else if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
            let rest = trimmed
                .strip_prefix("pub struct ")
                .unwrap_or_else(|| trimmed.strip_prefix("struct ").unwrap_or(""));
            let s_name = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches(['{', ';']);
            (DeclarationKind::Struct, format!("struct {s_name}"))
        } else if trimmed.starts_with("pub enum ") || trimmed.starts_with("enum ") {
            let rest = trimmed
                .strip_prefix("pub enum ")
                .unwrap_or_else(|| trimmed.strip_prefix("enum ").unwrap_or(""));
            let e_name = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches(['{', ';']);
            (DeclarationKind::Enum, format!("enum {e_name}"))
        } else if trimmed.starts_with("pub trait ") || trimmed.starts_with("trait ") {
            let rest = trimmed
                .strip_prefix("pub trait ")
                .unwrap_or_else(|| trimmed.strip_prefix("trait ").unwrap_or(""));
            let t_name = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches(['{', ';']);
            (DeclarationKind::Trait, format!("trait {t_name}"))
        } else if trimmed.starts_with("impl ") || trimmed.starts_with("impl<") {
            let rest = trimmed.strip_prefix("impl").unwrap_or("").trim();
            let impl_name = rest.split('{').next().unwrap_or("").trim();
            (DeclarationKind::TypeAlias, format!("impl {impl_name}"))
        } else if trimmed.starts_with("pub type ") || trimmed.starts_with("type ") {
            let rest = trimmed
                .strip_prefix("pub type ")
                .unwrap_or_else(|| trimmed.strip_prefix("type ").unwrap_or(""));
            let t_name = rest.split('=').next().unwrap_or("").trim();
            (DeclarationKind::TypeAlias, format!("type {t_name}"))
        } else if trimmed.starts_with("pub mod ") || trimmed.starts_with("mod ") {
            let rest = trimmed
                .strip_prefix("pub mod ")
                .unwrap_or_else(|| trimmed.strip_prefix("mod ").unwrap_or(""));
            let m_name = rest.split(';').next().unwrap_or("").trim();
            (DeclarationKind::Module, format!("mod {m_name}"))
        } else if trimmed.starts_with("macro_rules! ") {
            let m_name = trimmed
                .strip_prefix("macro_rules! ")
                .unwrap_or("")
                .split('{')
                .next()
                .unwrap_or("")
                .trim();
            (DeclarationKind::Macro, format!("macro_rules! {m_name}"))
        } else if trimmed.contains(".route(") {
            let route_line = trimmed
                .split(".route(")
                .nth(1)
                .unwrap_or("")
                .split(')')
                .next()
                .unwrap_or("")
                .trim();
            (DeclarationKind::Route, format!("route {route_line}"))
        } else {
            continue;
        };

        let (snippet, end_idx) = build_snippet(lines, idx, 10);
        decls.push(CodeDeclaration {
            file: file_path.to_string(),
            name,
            kind,
            line_start: idx + 1,
            line_end: end_idx,
            signature: trimmed.to_string(),
            snippet,
        });
    }

    decls
}

fn extract_python_declarations(file_path: &str, lines: &[&str]) -> Vec<CodeDeclaration> {
    let mut decls = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }

        let (kind, name) = if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
            let rest = trimmed
                .strip_prefix("async def ")
                .unwrap_or_else(|| trimmed.strip_prefix("def ").unwrap_or(""));
            let fn_name = rest.split('(').next().unwrap_or("").trim();
            (DeclarationKind::Function, format!("def {fn_name}"))
        } else if trimmed.starts_with("class ") {
            let rest = trimmed.strip_prefix("class ").unwrap_or("");
            let class_name = rest.split(['(', ':']).next().unwrap_or("").trim();
            (DeclarationKind::Class, format!("class {class_name}"))
        } else {
            continue;
        };

        let (snippet, end_idx) = build_snippet(lines, idx, 10);
        decls.push(CodeDeclaration {
            file: file_path.to_string(),
            name,
            kind,
            line_start: idx + 1,
            line_end: end_idx,
            signature: trimmed.to_string(),
            snippet,
        });
    }

    decls
}

fn extract_ts_js_declarations(file_path: &str, lines: &[&str]) -> Vec<CodeDeclaration> {
    let mut decls = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.is_empty() {
            continue;
        }

        let (kind, name) = if trimmed.starts_with("export function ")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("export async function ")
            || trimmed.starts_with("async function ")
        {
            let cleaned = trimmed
                .strip_prefix("export ")
                .unwrap_or(trimmed)
                .strip_prefix("async ")
                .unwrap_or(trimmed)
                .strip_prefix("function ")
                .unwrap_or(trimmed);
            let fn_name = cleaned.split('(').next().unwrap_or("").trim();
            (DeclarationKind::Function, format!("function {fn_name}"))
        } else if trimmed.starts_with("export class ") || trimmed.starts_with("class ") {
            let cleaned = trimmed
                .strip_prefix("export ")
                .unwrap_or(trimmed)
                .strip_prefix("class ")
                .unwrap_or("");
            let c_name = cleaned
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('{');
            (DeclarationKind::Class, format!("class {c_name}"))
        } else if trimmed.starts_with("export interface ") || trimmed.starts_with("interface ") {
            let cleaned = trimmed
                .strip_prefix("export ")
                .unwrap_or(trimmed)
                .strip_prefix("interface ")
                .unwrap_or("");
            let i_name = cleaned
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('{');
            (DeclarationKind::Interface, format!("interface {i_name}"))
        } else if trimmed.starts_with("export type ") || trimmed.starts_with("type ") {
            let cleaned = trimmed
                .strip_prefix("export ")
                .unwrap_or(trimmed)
                .strip_prefix("type ")
                .unwrap_or("");
            let t_name = cleaned.split('=').next().unwrap_or("").trim();
            (DeclarationKind::TypeAlias, format!("type {t_name}"))
        } else if (trimmed.starts_with("export const ") || trimmed.starts_with("const "))
            && (trimmed.contains("=>") || trimmed.contains("function"))
        {
            let cleaned = trimmed
                .strip_prefix("export ")
                .unwrap_or(trimmed)
                .strip_prefix("const ")
                .unwrap_or("");
            let c_name = cleaned.split(['=', ':']).next().unwrap_or("").trim();
            (DeclarationKind::Function, format!("const {c_name}"))
        } else {
            continue;
        };

        let (snippet, end_idx) = build_snippet(lines, idx, 10);
        decls.push(CodeDeclaration {
            file: file_path.to_string(),
            name,
            kind,
            line_start: idx + 1,
            line_end: end_idx,
            signature: trimmed.to_string(),
            snippet,
        });
    }

    decls
}

fn extract_go_declarations(file_path: &str, lines: &[&str]) -> Vec<CodeDeclaration> {
    let mut decls = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.is_empty() {
            continue;
        }

        let (kind, name) = if trimmed.starts_with("func ") {
            let rest = trimmed.strip_prefix("func ").unwrap_or("");
            let fn_name = rest.split('(').next().unwrap_or("").trim();
            (DeclarationKind::Function, format!("func {fn_name}"))
        } else if trimmed.starts_with("type ")
            && (trimmed.contains("struct") || trimmed.contains("interface"))
        {
            let rest = trimmed.strip_prefix("type ").unwrap_or("");
            let t_name = rest.split_whitespace().next().unwrap_or("").trim();
            let k = if trimmed.contains("struct") {
                DeclarationKind::Struct
            } else {
                DeclarationKind::Interface
            };
            (k, format!("type {t_name}"))
        } else {
            continue;
        };

        let (snippet, end_idx) = build_snippet(lines, idx, 10);
        decls.push(CodeDeclaration {
            file: file_path.to_string(),
            name,
            kind,
            line_start: idx + 1,
            line_end: end_idx,
            signature: trimmed.to_string(),
            snippet,
        });
    }

    decls
}

fn extract_markdown_headings(file_path: &str, lines: &[&str]) -> Vec<CodeDeclaration> {
    let mut decls = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let heading = trimmed.trim_start_matches('#').trim();
            let (snippet, end_idx) = build_snippet(lines, idx, 8);
            decls.push(CodeDeclaration {
                file: file_path.to_string(),
                name: heading.to_string(),
                kind: DeclarationKind::Heading,
                line_start: idx + 1,
                line_end: end_idx,
                signature: trimmed.to_string(),
                snippet,
            });
        }
    }

    decls
}

fn extract_generic_declarations(file_path: &str, lines: &[&str]) -> Vec<CodeDeclaration> {
    let mut decls = Vec::new();
    for (idx, line) in lines.iter().enumerate().take(30) {
        let trimmed = line.trim();
        if (trimmed.starts_with('[') && trimmed.ends_with(']')) || trimmed.ends_with(':') {
            let (snippet, end_idx) = build_snippet(lines, idx, 6);
            decls.push(CodeDeclaration {
                file: file_path.to_string(),
                name: trimmed.to_string(),
                kind: DeclarationKind::Other,
                line_start: idx + 1,
                line_end: end_idx,
                signature: trimmed.to_string(),
                snippet,
            });
        }
    }
    decls
}

/// Runs semantic codebase grep using Zev's zero-token, order-invariant decision engine
pub fn run_grep(engine: &ZevEngine, options: &GrepOptions) -> Result<GrepReport> {
    let start_time = Instant::now();

    // Stage 1: File Discovery via ignore::WalkBuilder respecting .gitignore
    let mut candidate_files = Vec::new();
    let walker = WalkBuilder::new(&options.path)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    for entry in walker.filter_map(std::result::Result::ok) {
        let p = entry.path();
        if p.is_file() && is_supported_source_file(p) {
            // Check file size (< 1MB)
            if let Ok(meta) = entry.metadata() {
                if meta.len() < 1_048_576 {
                    candidate_files.push(p.to_path_buf());
                    if candidate_files.len() >= options.max_files {
                        break;
                    }
                }
            }
        }
    }

    let files_scanned = candidate_files.len();

    // Stage 2: AST Declaration Extraction
    let mut all_declarations = Vec::new();
    for file_path in &candidate_files {
        if let Ok(content) = fs::read_to_string(file_path) {
            let relative_str = file_path
                .strip_prefix(&options.path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();
            let decls = extract_declarations(&relative_str, &content);
            all_declarations.extend(decls);
        }
    }

    let declarations_evaluated = all_declarations.len();

    // Stage 3: Semantic Relevance Scoring via ZevEngine
    let mut scored_matches = Vec::new();

    // If query is empty, return empty report
    if options.query.trim().is_empty() {
        return Ok(GrepReport {
            query: options.query.clone(),
            path: options.path.display().to_string(),
            files_scanned,
            declarations_evaluated,
            matches: Vec::new(),
            duration_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let query_lower = options.query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    for decl in all_declarations {
        // Build state representing the declaration candidate
        let state_repr = format!(
            "File: {}\nLine: {}\nKind: {}\nName: {}\nSignature: {}\nCode Snippet:\n{}",
            decl.file, decl.line_start, decl.kind, decl.name, decl.signature, decl.snippet
        );

        let question = Question::Boolean(BooleanQuestion {
            instructions: format!(
                "Does this code declaration implement, define, or relate to: {}?",
                options.query
            ),
            true_description: format!(
                "Yes. This code implements, references, or directly relates to {}.",
                options.query
            ),
            false_description: "No. This code is unrelated.".to_string(),
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });

        let mut questions = BTreeMap::new();
        questions.insert("relevance".to_string(), question);

        let req = ZevRequest {
            state: serde_json::Value::String(state_repr),
            questions,
            model: None,
            temperature: Some(engine.default_temperature),
            enable_temporal_facts: false,
            images: None,
        };

        // Evaluate in microsecond time
        if let Ok(resp) = engine.evaluate(&req) {
            if let Some(ans) = resp.answers.get("relevance") {
                let p_true = ans.probabilities.get("true").copied().unwrap_or(0.0);

                // Add heuristic boost if exact keyword matches occur in declaration name or signature
                let decl_name_lower = decl.name.to_lowercase();
                let decl_sig_lower = decl.signature.to_lowercase();
                let mut term_hits = 0;
                for term in &query_terms {
                    if decl_name_lower.contains(term) || decl_sig_lower.contains(term) {
                        term_hits += 1;
                    }
                }

                let boost = if !query_terms.is_empty() {
                    (term_hits as f64 / query_terms.len() as f64) * 0.4
                } else {
                    0.0
                };

                let final_score = (p_true + boost).min(1.0);

                if final_score >= options.threshold {
                    scored_matches.push(GrepMatch {
                        file: decl.file,
                        line_start: decl.line_start,
                        line_end: decl.line_end,
                        name: decl.name,
                        kind: decl.kind.to_string(),
                        signature: decl.signature,
                        score: final_score,
                        snippet: decl.snippet,
                    });
                }
            }
        }
    }

    // Sort by score descending, then by file/line
    scored_matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.file.cmp(&b.file))
            .then_with(|| a.line_start.cmp(&b.line_start))
    });

    if scored_matches.len() > options.limit {
        scored_matches.truncate(options.limit);
    }

    let duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok(GrepReport {
        query: options.query.clone(),
        path: options.path.display().to_string(),
        files_scanned,
        declarations_evaluated,
        matches: scored_matches,
        duration_ms,
    })
}

/// Formats the GrepReport into an ANSI-styled terminal output
pub fn format_human_report(report: &GrepReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\x1b[1;36mzev grep\x1b[0m: \"\x1b[1;37m{}\x1b[0m\" in \x1b[33m{}\x1b[0m\n",
        report.query, report.path
    ));
    out.push_str(&format!(
        "\x1b[90mScanned {} files, evaluated {} declarations in {:.2}ms\x1b[0m\n\n",
        report.files_scanned, report.declarations_evaluated, report.duration_ms
    ));

    if report.matches.is_empty() {
        out.push_str("\x1b[33mNo matching declarations found above threshold.\x1b[0m\n");
        return out;
    }

    for (idx, m) in report.matches.iter().enumerate() {
        let score_pct = (m.score * 100.0).round() as usize;
        let score_color = if score_pct >= 80 {
            "\x1b[1;32m" // bold green
        } else if score_pct >= 50 {
            "\x1b[1;33m" // bold yellow
        } else {
            "\x1b[1;90m" // gray
        };

        out.push_str(&format!(
            "{:>2}. {score_color}[{:>2}%]\x1b[0m \x1b[1m{}:{}:{}\x1b[0m (\x1b[35m{}\x1b[0m: \x1b[34m{}\x1b[0m)\n",
            idx + 1,
            score_pct,
            m.file,
            m.line_start,
            m.line_end,
            m.kind,
            m.name
        ));

        // Print snippet indented
        for line in m.snippet.lines().take(4) {
            out.push_str(&format!("      \x1b[90m│\x1b[0m {}\n", line));
        }
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_rust_declarations() {
        let code = r#"
pub struct DecisionEngine {
    pub inner: ZevEngine,
}

impl DecisionEngine {
    pub fn new() -> Self {
        Self { inner: ZevEngine::default() }
    }

    pub async fn evaluate(&self) -> bool {
        true
    }
}

pub enum WireQuestion {
    Noul,
    Choice,
}
"#;
        let decls = extract_declarations("src/test.rs", code);
        assert!(!decls.is_empty());
        assert!(decls.iter().any(|d| d.name.contains("DecisionEngine")));
        assert!(decls.iter().any(|d| d.name.contains("new")));
        assert!(decls.iter().any(|d| d.name.contains("WireQuestion")));
    }

    #[test]
    fn test_extract_ts_declarations() {
        let code = r#"
export interface UserConfig {
    id: string;
}

export function createServer(port: number) {
    return port;
}

export const handler = async (req: Request) => {
    return req;
};
"#;
        let decls = extract_declarations("src/test.ts", code);
        assert!(!decls.is_empty());
        assert!(decls.iter().any(|d| d.name.contains("UserConfig")));
        assert!(decls.iter().any(|d| d.name.contains("createServer")));
        assert!(decls.iter().any(|d| d.name.contains("handler")));
    }

    #[test]
    fn test_run_grep_basic() {
        let engine = ZevEngine::default();
        let opts = GrepOptions {
            query: "router endpoint".to_string(),
            path: PathBuf::from("src"),
            limit: 5,
            threshold: 0.2,
            json: false,
            include_evidence: true,
            max_files: 50,
        };
        let report = run_grep(&engine, &opts).expect("run grep");
        assert!(report.files_scanned > 0);
        assert!(report.declarations_evaluated > 0);
    }
}
