//! Host-side token sampling (temperature / top-k / top-p) with a small xorshift RNG.

#[derive(Clone, Debug)]
pub struct Params {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Params {
    /// Qwen3.5 recommended sampling (general.sampling.* in the GGUF).
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            seed: 0x5eed,
        }
    }
}

impl Params {
    /// Greedy decoding is selected by either zero (or negative) temperature or `top_k == 1`.
    /// The decode loop and `Sampler::sample` must agree on this rule.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

pub struct Sampler {
    p: Params,
    state: u64,
}

impl Sampler {
    pub fn new(p: Params) -> Self {
        let state = p.seed.max(1);
        Self { p, state }
    }

    pub fn is_greedy(&self) -> bool {
        self.p.is_greedy()
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let argmax = || crate::vec::backend().argmax(logits) as u32;
        if self.p.is_greedy() {
            return argmax();
        }
        let k = if self.p.top_k == 0 {
            logits.len()
        } else {
            self.p.top_k.min(logits.len())
        };
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        if k < idx.len() {
            idx.select_nth_unstable_by(k, |&a, &b| {
                logits[b as usize].total_cmp(&logits[a as usize])
            });
            idx.truncate(k);
        }
        idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        let mx = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| ((logits[i as usize] - mx) / self.p.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        let mut cut = probs.len();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= self.p.top_p {
                cut = i + 1;
                break;
            }
        }
        let total: f32 = probs[..cut].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..cut {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[cut - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_rule() {
        let base = Params::default();
        let greedy = |temperature, top_k| Params {
            temperature,
            top_k,
            ..base.clone()
        };
        assert!(greedy(1.0, 1).is_greedy());
        assert!(greedy(0.0, 20).is_greedy());
        assert!(greedy(-0.5, 20).is_greedy());
        assert!(!greedy(0.7, 20).is_greedy());
        assert!(!greedy(1.0, 0).is_greedy());
    }

    #[test]
    fn greedy_sample_picks_argmax() {
        let p = Params {
            temperature: 0.0,
            ..Params::default()
        };
        assert_eq!(Sampler::new(p).sample(&[0.1, -1.0, 3.0, 2.0]), 2);
    }
}
