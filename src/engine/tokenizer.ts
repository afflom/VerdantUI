/**
 * GGUF LLaMA SentencePiece and GPT-2 Byte-Level BPE tokenizer matching llama.cpp behavior.
 *
 * Supports:
 * - SentencePiece (`llama`): `▁` space normalization, score-driven BPE merges, `<0xXX>` byte fallbacks.
 * - Byte-Level BPE (`gpt2`, used by SmolLM2, Qwen2.5, Llama-3): OpenAI `bytes_to_unicode()` bijection,
 *   atomic special/control token splitting, pre-tokenization regex, merge-rank BPE, and streaming UTF-8 decode.
 */

import type { ParsedGgufModel } from './gguf';

const SPIECE_UNDERLINE = '\u2581'; // '▁'
const utf8Encoder = new TextEncoder();
const utf8Decoder = new TextDecoder('utf-8', { fatal: false });

/** OpenAI GPT-2 byte-to-unicode bijection (256 entries) used by SmolLM2, Qwen2.5, Llama-3, and GPT-2. */
const GPT2_BYTE_TO_CHAR: readonly string[] = (() => {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b); // '!'..'~'
  for (let b = 161; b <= 172; b++) bs.push(b); // '¡'..'¬'
  for (let b = 174; b <= 255; b++) bs.push(b); // '®'..'ÿ'
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const table = new Array<string>(256);
  for (let i = 0; i < bs.length; i++) {
    table[bs[i]] = String.fromCharCode(cs[i]);
  }
  return table;
})();

const GPT2_CHAR_TO_BYTE: ReadonlyMap<string, number> = (() => {
  const map = new Map<string, number>();
  for (let b = 0; b < 256; b++) {
    map.set(GPT2_BYTE_TO_CHAR[b], b);
  }
  return map;
})();

/** Standard Llama-3 / SmolLM2 / Qwen2 byte-level BPE pre-tokenization pattern. */
const GPT2_PRETOKENIZE_RE =
  /(?:'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

/** SmolLM2 uses single-digit number splitting (`\p{N}`) while Qwen2 / Llama-3 split 1..3 digits or 1 digit. */
const SMOLLM_PRETOKENIZE_RE =
  /(?:'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const KNOWN_EOG_PIECES = new Set<string>([
  '<|im_end|>',
  '<|endoftext|>',
  '<|eot_id|>',
  '<|eom_id|>',
  '<|end_of_text|>',
  '<|end|>',
  '<end_of_turn>',
  '</s>',
]);

export class GgufTokenizer {
  readonly vocab: readonly string[];
  readonly scores: readonly number[];
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly unkTokenId: number;
  readonly addBosToken: boolean;
  readonly modelType: string;
  readonly preTokenizer: string;
  readonly eogTokenIds: ReadonlySet<number>;

  private readonly pieceToId = new Map<string, number>();
  private readonly byteToId = new Map<number, number>();
  private readonly mergeRank = new Map<string, number>();
  private readonly controlTokenIds = new Set<number>();
  private readonly specialTokenRegex: RegExp | null = null;

  constructor(model: ParsedGgufModel) {
    const rawTokens = model.metadata['tokenizer.ggml.tokens'];
    if (!Array.isArray(rawTokens) || rawTokens.length === 0) {
      throw new Error('GGUF model is missing "tokenizer.ggml.tokens" vocabulary.');
    }

    const vocab: string[] = rawTokens.map((t) => String(t));
    const rawScores = model.metadata['tokenizer.ggml.scores'];
    const scores: number[] = Array.isArray(rawScores)
      ? rawScores.map((s) => (typeof s === 'number' ? s : 0))
      : new Array<number>(vocab.length).fill(0);

    const rawTypes = model.metadata['tokenizer.ggml.token_type'];
    const tokenTypes: number[] | null = Array.isArray(rawTypes)
      ? rawTypes.map((t) => (typeof t === 'number' ? t : 1))
      : null;

    this.vocab = vocab;
    this.scores = scores;
    this.bosTokenId = model.hparams.bosTokenId;
    this.eosTokenId = model.hparams.eosTokenId;
    this.unkTokenId = model.hparams.unkTokenId;
    this.addBosToken = model.hparams.addBosToken;
    this.modelType = model.hparams.tokenizerModel;
    this.preTokenizer = model.hparams.preTokenizer;

    const specialPieces: string[] = [];
    const eogSet = new Set<number>();
    if (this.eosTokenId >= 0 && this.eosTokenId < vocab.length) {
      eogSet.add(this.eosTokenId);
    }

    const eotMeta = model.metadata['tokenizer.ggml.eot_token_id'];
    if (typeof eotMeta === 'number' && eotMeta >= 0 && eotMeta < vocab.length) {
      eogSet.add(eotMeta);
    }

    for (let i = 0; i < vocab.length; i++) {
      const piece = vocab[i];
      if (!this.pieceToId.has(piece)) {
        this.pieceToId.set(piece, i);
      }

      // SentencePiece byte fallback tokens: <0x00> .. <0xFF>
      if (piece.length === 6 && piece.startsWith('<0x') && piece.endsWith('>')) {
        const byteVal = parseInt(piece.slice(3, 5), 16);
        if (!Number.isNaN(byteVal) && byteVal >= 0 && byteVal <= 255) {
          this.byteToId.set(byteVal, i);
        }
      }

      const tType = tokenTypes ? tokenTypes[i] : 1;
      if (tType === 3 || tType === 4) {
        if (tType === 3) {
          this.controlTokenIds.add(i);
        }
        if (piece.length > 0) {
          specialPieces.push(piece);
        }
      }

      if (KNOWN_EOG_PIECES.has(piece)) {
        eogSet.add(i);
        this.controlTokenIds.add(i);
      }
    }

    // Ensure BOS / EOS are treated as control tokens during display decoding
    if (this.bosTokenId >= 0 && this.bosTokenId < vocab.length) {
      this.controlTokenIds.add(this.bosTokenId);
    }
    if (this.eosTokenId >= 0 && this.eosTokenId < vocab.length) {
      this.controlTokenIds.add(this.eosTokenId);
    }

    this.eogTokenIds = eogSet;

    if (specialPieces.length > 0) {
      // Match longest special tokens first so `<|im_start|>` is captured atomically
      const uniqueSpecials = Array.from(new Set(specialPieces)).sort((a, b) => b.length - a.length);
      this.specialTokenRegex = new RegExp(`(${uniqueSpecials.map(escapeRegex).join('|')})`, 'g');
    }

    const rawMerges = model.metadata['tokenizer.ggml.merges'];
    if (Array.isArray(rawMerges)) {
      for (let i = 0; i < rawMerges.length; i++) {
        this.mergeRank.set(String(rawMerges[i]), i);
      }
    }
  }

  /** Returns whether this tokenizer uses GPT-2 byte-level BPE (SmolLM2, Qwen2.5, Llama-3). */
  get isByteLevelBpe(): boolean {
    return this.modelType === 'gpt2' || (this.mergeRank.size > 0 && this.modelType !== 'llama');
  }

  encode(text: string, addBos = this.addBosToken): number[] {
    const tokens: number[] = [];
    if (addBos && this.bosTokenId >= 0 && this.bosTokenId < this.vocab.length) {
      tokens.push(this.bosTokenId);
    }
    if (!text) {
      return tokens;
    }

    if (this.isByteLevelBpe) {
      this.encodeByteLevelBpe(text, tokens);
      return tokens;
    }

    // SentencePiece path: split around special tokens if present, otherwise normalize whole string
    if (this.specialTokenRegex) {
      let lastIndex = 0;
      this.specialTokenRegex.lastIndex = 0;
      let match: RegExpExecArray | null;
      let isFirstSegment = true;
      while ((match = this.specialTokenRegex.exec(text)) !== null) {
        if (match.index > lastIndex) {
          this.encodeSentencePieceSegment(text.slice(lastIndex, match.index), isFirstSegment, tokens);
          isFirstSegment = false;
        }
        const specialId = this.pieceToId.get(match[0]);
        if (specialId !== undefined) {
          tokens.push(specialId);
        }
        isFirstSegment = false;
        lastIndex = match.index + match[0].length;
      }
      if (lastIndex < text.length) {
        this.encodeSentencePieceSegment(text.slice(lastIndex), isFirstSegment, tokens);
      }
      return tokens;
    }

    this.encodeSentencePieceSegment(text, true, tokens);
    return tokens;
  }

  private encodeByteLevelBpe(text: string, outTokens: number[]): void {
    const preRe =
      this.preTokenizer === 'smollm' || this.preTokenizer === 'qwen2'
        ? SMOLLM_PRETOKENIZE_RE
        : GPT2_PRETOKENIZE_RE;

    const encodePlainChunk = (segment: string) => {
      if (!segment) return;
      preRe.lastIndex = 0;
      const matches = segment.match(preRe);
      if (!matches) return;

      for (const word of matches) {
        const bytes = utf8Encoder.encode(word);
        const pieces: string[] = new Array<string>(bytes.length);
        for (let i = 0; i < bytes.length; i++) {
          pieces[i] = GPT2_BYTE_TO_CHAR[bytes[i]];
        }

        // Fast exact match shortcut when the entire pre-tokenized piece is already in vocab
        if (pieces.length > 1) {
          const joined = pieces.join('');
          const directId = this.pieceToId.get(joined);
          if (directId !== undefined && this.mergeRank.size === 0) {
            outTokens.push(directId);
            continue;
          }
        }

        // Standard GPT-2 BPE merge loop: repeatedly merge the lowest-rank adjacent pair
        while (pieces.length >= 2) {
          let bestRank = Infinity;
          let bestIdx = -1;
          for (let i = 0; i < pieces.length - 1; i++) {
            const rank = this.mergeRank.get(`${pieces[i]} ${pieces[i + 1]}`);
            if (rank !== undefined && rank < bestRank) {
              bestRank = rank;
              bestIdx = i;
            }
          }
          if (bestIdx === -1) break;
          pieces.splice(bestIdx, 2, pieces[bestIdx] + pieces[bestIdx + 1]);
        }

        for (const p of pieces) {
          const id = this.pieceToId.get(p);
          if (id !== undefined) {
            outTokens.push(id);
          } else {
            // Byte fallback if a merged piece is missing from vocab
            for (const ch of p) {
              const cid = this.pieceToId.get(ch);
              outTokens.push(cid !== undefined ? cid : this.unkTokenId);
            }
          }
        }
      }
    };

    if (!this.specialTokenRegex) {
      encodePlainChunk(text);
      return;
    }

    let lastIndex = 0;
    this.specialTokenRegex.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = this.specialTokenRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        encodePlainChunk(text.slice(lastIndex, match.index));
      }
      const specialId = this.pieceToId.get(match[0]);
      if (specialId !== undefined) {
        outTokens.push(specialId);
      }
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
      encodePlainChunk(text.slice(lastIndex));
    }
  }

  private encodeSentencePieceSegment(
    text: string,
    prependSpace: boolean,
    outTokens: number[],
  ): void {
    if (!text) return;
    const normalized =
      this.modelType === 'llama'
        ? (prependSpace && !text.startsWith(' ') ? SPIECE_UNDERLINE : '') +
          text.replace(/ /g, SPIECE_UNDERLINE)
        : text;

    const symbols: number[] = [];
    for (const ch of normalized) {
      const id = this.pieceToId.get(ch);
      if (id !== undefined) {
        symbols.push(id);
      } else {
        const bytes = utf8Encoder.encode(ch);
        for (const b of bytes) {
          const byteId = this.byteToId.get(b);
          symbols.push(byteId !== undefined ? byteId : this.unkTokenId);
        }
      }
    }

    while (symbols.length >= 2) {
      let bestIdx = -1;
      let bestId = -1;
      let bestScore = -Infinity;

      for (let i = 0; i < symbols.length - 1; i++) {
        const left = this.vocab[symbols[i]];
        const right = this.vocab[symbols[i + 1]];
        const merged = left + right;
        const mergedId = this.pieceToId.get(merged);
        if (mergedId === undefined) continue;

        let score: number;
        if (this.mergeRank.size > 0) {
          const rank = this.mergeRank.get(`${left} ${right}`);
          if (rank === undefined) continue;
          score = -rank;
        } else {
          score = this.scores[mergedId];
          if (score === 0) {
            score = -mergedId * 1e-6;
          }
        }

        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
          bestId = mergedId;
        }
      }

      if (bestIdx === -1) break;
      symbols.splice(bestIdx, 2, bestId);
    }

    outTokens.push(...symbols);
  }

  /**
   * Stateful streaming decoder handling both:
   * - GPT-2 byte-level BPE (`Ġ` -> space, `Ċ` -> newline, arbitrary multi-byte UTF-8 across tokens)
   * - SentencePiece `<0xXX>` byte-fallback tokens and `▁` space markers
   */
  createStreamDecoder(): {
    push: (tokenId: number) => string;
    flush: () => string;
  } {
    const pendingBytes: number[] = [];
    let isFirst = true;
    let hasStreamBytes = false;
    const streamDecoder = new TextDecoder('utf-8', { fatal: false });
    const byteLevel = this.isByteLevelBpe;

    const flushPending = (final: boolean): string => {
      if (pendingBytes.length === 0 && !(final && hasStreamBytes)) return '';
      const bytes = new Uint8Array(pendingBytes);
      pendingBytes.length = 0;
      const decoded = streamDecoder.decode(bytes, { stream: !final });
      hasStreamBytes = !final;
      if (!byteLevel && isFirst && decoded.length > 0) {
        isFirst = false;
        return decoded.startsWith(' ') ? decoded.slice(1) : decoded;
      }
      if (decoded.length > 0) {
        isFirst = false;
      }
      return decoded;
    };

    return {
      push: (tokenId: number): string => {
        if (tokenId < 0 || tokenId >= this.vocab.length) return '';
        if (this.controlTokenIds.has(tokenId)) return '';

        const piece = this.vocab[tokenId];

        if (byteLevel) {
          for (let i = 0; i < piece.length; i++) {
            const b = GPT2_CHAR_TO_BYTE.get(piece[i]);
            if (b !== undefined) {
              pendingBytes.push(b);
            } else {
              const extra = utf8Encoder.encode(piece[i]);
              for (let k = 0; k < extra.length; k++) pendingBytes.push(extra[k]);
            }
          }
          return flushPending(false);
        }

        if (piece.length === 6 && piece.startsWith('<0x') && piece.endsWith('>')) {
          const byteVal = parseInt(piece.slice(3, 5), 16);
          if (!Number.isNaN(byteVal)) {
            pendingBytes.push(byteVal);
            return flushPending(false);
          }
        }

        const prefix = flushPending(true);
        let text = piece.replace(/\u2581/g, ' ');
        if (isFirst && prefix.length === 0 && text.length > 0) {
          isFirst = false;
          if (text.startsWith(' ')) text = text.slice(1);
        } else if (text.length > 0) {
          isFirst = false;
        }
        return prefix + text;
      },
      flush: (): string => flushPending(true),
    };
  }

  decodeToken(tokenId: number, isFirstOutputToken = false): string {
    if (tokenId < 0 || tokenId >= this.vocab.length) {
      return '';
    }
    if (this.controlTokenIds.has(tokenId)) {
      return '';
    }
    const piece = this.vocab[tokenId];
    if (this.isByteLevelBpe) {
      const bytes: number[] = [];
      for (let i = 0; i < piece.length; i++) {
        const b = GPT2_CHAR_TO_BYTE.get(piece[i]);
        if (b !== undefined) {
          bytes.push(b);
        } else {
          for (const ub of utf8Encoder.encode(piece[i])) bytes.push(ub);
        }
      }
      return utf8Decoder.decode(new Uint8Array(bytes));
    }
    if (piece.length === 6 && piece.startsWith('<0x') && piece.endsWith('>')) {
      const byteVal = parseInt(piece.slice(3, 5), 16);
      if (!Number.isNaN(byteVal)) {
        return utf8Decoder.decode(new Uint8Array([byteVal]));
      }
    }
    let text = piece.replace(/\u2581/g, ' ');
    if (isFirstOutputToken && text.startsWith(' ')) {
      text = text.slice(1);
    }
    return text;
  }

  decode(tokenIds: readonly number[]): string {
    const decoder = this.createStreamDecoder();
    let out = '';
    for (let i = 0; i < tokenIds.length; i++) {
      out += decoder.push(tokenIds[i]);
    }
    out += decoder.flush();
    return out;
  }
}
