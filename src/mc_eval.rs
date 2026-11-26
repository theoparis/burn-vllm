//! Multiple-choice log-likelihood scoring for Inkling-2, following the protocol used by
//! the lm-evaluation-harness in arXiv:2610.02185 (App. B.2): each candidate continuation is
//! scored by the sum of its token log-probabilities, and the highest score wins.
//!
//! The dataset is a JSON array of `{"q": ..., "choices": [...], "answer": <index>}`.
//! Each question's prompt is fed once; the resulting state is cloned per candidate so the
//! answer continuations are scored from an identical prefix.

use crate::gguf::Gguf;
use crate::inkling2;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
struct Question {
    q: String,
    choices: Vec<String>,
    answer: usize,
}

/// Log-softmax of guided (or unguided) logits, with max subtraction for stability.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    let log_sum = sum.ln();
    logits.iter().map(|&v| v - max - log_sum).collect()
}

fn score_choice(
    model: &inkling2::Model,
    st: &inkling2::State,
    continuation: &[u32],
    first_logits: &[f32],
) -> Result<(f32, usize)> {
    let mut st = st.clone();
    let mut logits = first_logits.to_vec();
    let mut total = 0.0f32;
    for &tok in continuation {
        let lp = log_softmax(&logits)[tok as usize];
        ensure!(!lp.is_nan(), "NaN log-prob for token {tok}");
        total += lp;
        logits = model.forward(&mut st, tok)?;
    }
    Ok((total, continuation.len()))
}

pub fn run(gguf: &Gguf, path: &std::path::Path, limit: usize) -> Result<()> {
    let tok = Tokenizer::from_gguf(gguf)?;
    let model = inkling2::Model::new(gguf)?;
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("reading MC dataset {}", path.display()))?;
    let questions: Vec<Question> =
        serde_json::from_str(&data).context("parsing MC dataset JSON")?;
    ensure!(!questions.is_empty(), "MC dataset is empty");

    let n_questions = if limit > 0 {
        questions.len().min(limit)
    } else {
        questions.len()
    };
    let done = std::sync::atomic::AtomicUsize::new(0);
    let mode = match std::env::var("LOOPCD").ok().as_deref() {
        Some("hidden") => format!(
            "hidden ω={}",
            std::env::var("LOOPCD_OMEGA").unwrap_or_else(|_| "0.25".into())
        ),
        Some("logits") => format!(
            "logits ω={}{}",
            std::env::var("LOOPCD_OMEGA").unwrap_or_else(|_| "0.25".into()),
            if std::env::var_os("LOOPCD_ADAPTIVE").is_some() {
                " adaptive"
            } else {
                ""
            }
        ),
        _ => "unguided".into(),
    };
    eprintln!("mc_eval: {n_questions} questions, mode={mode}");

    use rayon::prelude::*;
    let results: Result<Vec<(usize, usize)>> = questions
        .par_iter()
        .take(n_questions)
        .enumerate()
        .map(|(qi, q)| -> Result<(usize, usize)> {
            ensure!(
                q.answer < q.choices.len(),
                "question {qi}: bad answer index"
            );
            let prompt = format!("Question: {}\nAnswer:", q.q);
            let prompt_ids = tok.encode(&prompt)?;
            ensure!(!prompt_ids.is_empty(), "question {qi}: empty prompt");

            let mut st = inkling2::State::new(model.config());
            let mut first_logits = Vec::new();
            for &t in &prompt_ids {
                first_logits = model.forward(&mut st, t)?;
            }

            let mut scores = Vec::with_capacity(q.choices.len());
            for choice in &q.choices {
                let cont = tok.encode(&format!(" {}", choice.trim()))?;
                ensure!(!cont.is_empty(), "question {qi}: empty choice {choice:?}");
                scores.push(score_choice(&model, &st, &cont, &first_logits)?);
            }

            let best = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.0.total_cmp(&b.1.0))
                .map(|(i, _)| i)
                .expect("at least one choice");
            let best_norm = scores
                .iter()
                .enumerate()
                .max_by(|a, b| (a.1.0 / a.1.1 as f32).total_cmp(&(b.1.0 / b.1.1 as f32)))
                .map(|(i, _)| i)
                .expect("at least one choice");
            let finished = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if finished % 10 == 0 || finished == n_questions {
                eprintln!("mc_eval: {finished}/{n_questions} done");
            }
            Ok((
                usize::from(best == q.answer),
                usize::from(best_norm == q.answer),
            ))
        })
        .collect();
    let results = results?;
    let correct: usize = results.iter().map(|r| r.0).sum();
    let norm_correct: usize = results.iter().map(|r| r.1).sum();

    println!(
        "mc_eval mode={mode} n={} acc={:.1}% acc_norm={:.1}%",
        results.len(),
        100.0 * correct as f64 / results.len() as f64,
        100.0 * norm_correct as f64 / results.len() as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_matches_reference() {
        let logits = [2.0f32, 1.0, 0.1, -3.0];
        let got = log_softmax(&logits);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom: f64 = logits.iter().map(|&v| (f64::from(v - max)).exp()).sum();
        for (&v, &g) in logits.iter().zip(&got) {
            let want = f64::from(v - max) - denom.ln();
            assert!((f64::from(g) - want).abs() < 1e-5, "got {g}, want {want}");
        }
        let sum: f32 = got.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {sum}");
    }
}
