//! Byte-level BPE tokenizer built from GGUF `tokenizer.ggml.*` metadata (gpt2 model,
//! qwen35/hy_v4 pre-tokenizers).

use crate::gguf::Gguf;
use anyhow::{Context, Result, bail};
use fancy_regex::Regex;
use std::collections::HashMap;

const QWEN35_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Transcribed from llama.cpp `src/llama-vocab.cpp`, `LLAMA_VOCAB_PRE_TYPE_HY_V4`
// (shared with deepseek3-llm / hunyuan-dense / joyai-llm). llama.cpp applies these
// three expressions in order, splitting the pieces produced by the previous one.
// 1) numbers in groups of up to three digits,
// 2) CJK runs (literal ranges, unlike the \p{...} expressions),
// 3) letters with an optional leading punctuation/non-letter, punctuation, newlines,
//    whitespace runs.
const HY_V4_DIGITS: &str = r"\p{N}{1,3}";
const HY_V4_CJK: &str = r"[一-龥぀-ゟ゠-ヿ]+";
const HY_V4_MAIN: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]\^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

pub struct Tokenizer {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Special (control / user-defined) tokens, longest first, matched verbatim.
    specials: Vec<(String, u32)>,
    is_special: Vec<bool>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    /// Pre-tokenizer expressions, applied in order (llama.cpp semantics).
    pre: Vec<Regex>,
    /// GPT-2 BPE with `ignore_merges`: a pre-token present verbatim in the vocab is
    /// emitted as one token instead of being split into byte symbols and merged.
    ignore_merges: bool,
    pub eos: u32,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g.meta_str("tokenizer.ggml.model")?;
        if model != "gpt2" {
            bail!("unsupported tokenizer model {model}");
        }
        let pre = g.meta_str("tokenizer.ggml.pre").unwrap_or("default");
        // inkling2 uses a GPT-2 regex pre-tokenizer; the Qwen35 regex is close enough for
        // ordinary text, but its merges must be skipped when a piece is a vocab token.
        let ignore_merges = pre == "inkling2";
        // Hy4 (hy_v4) uses the deepseek3/hunyuan-dense regex set from llama.cpp.
        // Its BPE does not use `ignore_merges`, matching our default.
        let pre_patterns: &[&str] = if pre == "hy_v4" {
            &[HY_V4_DIGITS, HY_V4_CJK, HY_V4_MAIN]
        } else {
            if pre != "qwen35" && pre != "qwen2" && !ignore_merges {
                eprintln!("warning: pre-tokenizer {pre:?} not recognized, using qwen35 regex");
            }
            &[QWEN35_PRE]
        };
        let pre_regexes = pre_patterns
            .iter()
            .map(|p| Regex::new(p).context("invalid pre-tokenizer regex"))
            .collect::<Result<Vec<_>>>()?;
        let tokens: Vec<String> = g
            .meta_array("tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).context("token not a string"))
            .collect::<Result<_>>()?;
        let types: Vec<u64> = g
            .meta_array("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(1))
            .collect();
        let vocab: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let mut merges = HashMap::new();
        for (rank, m) in g.meta_array("tokenizer.ggml.merges")?.iter().enumerate() {
            let m = m.as_str().context("merge not a string")?;
            let (a, b) = m.split_once(' ').context("bad merge")?;
            let (Some(&ia), Some(&ib), Some(&im)) =
                (vocab.get(a), vocab.get(b), vocab.get(&format!("{a}{b}")))
            else {
                continue;
            };
            merges.entry((ia, ib)).or_insert((rank as u32, im));
        }
        let is_special: Vec<bool> = types.iter().map(|&t| t == 3 || t == 4).collect();
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| is_special[*i] && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let byte_to_char = bytes_to_unicode();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let eos = g.meta_u64("tokenizer.ggml.eos_token_id")? as u32;
        Ok(Self {
            tokens,
            vocab,
            merges,
            specials,
            is_special,
            byte_to_char,
            char_to_byte,
            pre: pre_regexes,
            ignore_merges,
            eos,
        })
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Earliest special-token occurrence (longest wins at equal positions).
            let mut best: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(p) = rest.find(s.as_str())
                    && best.is_none_or(|(bp, bs, _)| p < bp || (p == bp && s.len() > bs.len()))
                {
                    best = Some((p, s, *id));
                }
            }
            let (plain, special) = match best {
                Some((p, s, id)) => (&rest[..p], Some((s.len(), id))),
                None => (rest, None),
            };
            self.encode_plain(plain, &mut out)?;
            match special {
                Some((len, id)) => {
                    out.push(id);
                    rest = &rest[plain.len() + len..];
                }
                None => break,
            }
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        // llama.cpp applies every pre-tokenizer expression in order, further splitting
        // the pieces produced by the previous one (unmatched gaps are kept as pieces).
        let mut pieces = vec![text];
        for re in &self.pre {
            let mut split = Vec::new();
            for piece in pieces {
                let mut last = 0;
                for m in re.find_iter(piece) {
                    let m = m?;
                    if m.start() > last {
                        split.push(&piece[last..m.start()]);
                    }
                    split.push(&piece[m.start()..m.end()]);
                    last = m.end();
                }
                if last < piece.len() {
                    split.push(&piece[last..]);
                }
            }
            pieces = split;
        }
        for piece in pieces {
            if self.ignore_merges
                && let Some(&id) = self.vocab.get(piece)
            {
                out.push(id);
                continue;
            }
            let mut syms: Vec<u32> = piece
                .bytes()
                .map(|b| {
                    let c = self.byte_to_char[b as usize];
                    self.vocab
                        .get(c.encode_utf8(&mut [0; 4]) as &str)
                        .copied()
                        .context("byte symbol missing from vocab")
                })
                .collect::<Result<_>>()?;
            loop {
                let best = syms
                    .windows(2)
                    .enumerate()
                    .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(r, m)| (r, i, m)))
                    .min();
                let Some((_, i, m)) = best else { break };
                syms[i] = m;
                syms.remove(i + 1);
            }
            out.extend_from_slice(&syms);
        }
        Ok(())
    }

    /// Raw bytes for a token (special tokens render as their literal text).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.is_special[id as usize] {
            return t.as_bytes().to_vec();
        }
        t.chars()
            .map(|c| self.char_to_byte.get(&c).copied().unwrap_or(b'?'))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HY4_MODEL: &str = "/mnt/models/hy4-flash-gguf/Hy4-flash-Q4_K_M-00001-of-00002.gguf";

    fn hy4_tokenizer() -> Option<Tokenizer> {
        let path = std::path::Path::new(HY4_MODEL);
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return None;
        }
        let g = Gguf::open(path).unwrap();
        assert_eq!(g.meta_str("tokenizer.ggml.pre").unwrap(), "hy_v4");
        Some(Tokenizer::from_gguf(&g).unwrap())
    }

    #[test]
    fn hy4_prompt_matches_llama_cpp_oracle() {
        let Some(tok) = hy4_tokenizer() else { return };
        // llama.cpp oracle (`tokenize_hy4`): 12433,11,1215,2046,341
        assert_eq!(
            tok.encode("Hello, my name is").unwrap(),
            [12433, 11, 1215, 2046, 341]
        );
    }

    #[test]
    fn hy4_pre_tokenizer_matches_llama_cpp_oracle() {
        let Some(tok) = hy4_tokenizer() else { return };
        // Token ids produced by the llama.cpp oracle for the same GGUF. In particular
        // digits are grouped 1-3 at a time (`\p{N}{1,3}`), unlike the qwen35 fallback.
        for (text, expected) in [
            ("1234", vec![6636, 19]),
            ("1234567", vec![6636, 17243, 22]),
            ("1234567890", vec![6636, 17243, 29041, 15]),
            ("abc123def", vec![22354, 6636, 2345]),
            (",my name", vec![11, 4621, 2046]),
            (
                "Hello, world! It's 2026.",
                vec![12433, 11, 3048, 0, 1177, 715, 220, 983, 21, 13],
            ),
            ("café", vec![66, 2592, 729]),
            (
                "naïve café — 東京 🚀",
                vec![3231, 99726, 75970, 2227, 220, 37248, 96542, 222],
            ),
            ("  indented\ttab", vec![220, 1189, 31471, 197, 20484]),
            ("foo_bar", vec![17726, 64006]),
            ("It's 2026.", vec![2294, 715, 220, 983, 21, 13]),
            (
                "一龥぀ゟ゠ヿ",
                vec![398, 3275, 98, 408, 222, 581, 253, 581, 254, 690, 123],
            ),
            ("Hello,world", vec![12433, 11, 25509]),
            ("  hello   world", vec![220, 33146, 256, 3048]),
        ] {
            let ids = tok.encode(text).unwrap();
            assert_eq!(ids, expected, "oracle mismatch for {text:?}");
            let bytes: Vec<u8> = ids.iter().flat_map(|&i| tok.token_bytes(i)).collect();
            assert_eq!(String::from_utf8(bytes).unwrap(), text);
        }
    }

    #[test]
    fn roundtrip() {
        let path = std::path::Path::new(crate::DEFAULT_MODEL);
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let g = Gguf::open(path).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        for text in [
            "Hello, world! It's 2026.",
            "<|im_start|>user\nnaïve café — 東京 🚀\n\n  indented\ttab<|im_end|>\n",
            "1234567 x=y+z;\r\n",
        ] {
            let ids = tok.encode(text).unwrap();
            let bytes: Vec<u8> = ids.iter().flat_map(|&i| tok.token_bytes(i)).collect();
            assert_eq!(String::from_utf8(bytes).unwrap(), text);
        }
        let ids = tok.encode("<|im_start|>user").unwrap();
        assert_eq!(ids[0], tok.token_id("<|im_start|>").unwrap());
        // Digits split one per token (\p{N} alternative).
        assert_eq!(tok.encode("123").unwrap().len(), 3);
    }
}
