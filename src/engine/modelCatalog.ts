/**
 * Curated GGUF model catalog compatible with llama.cpp + afflom/vTensor.
 * Includes practical instruction-tuned chat models (SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B)
 * plus the bundled offline stories260K.gguf and afflom/vTensor stories15M conformance reference.
 */

export interface CatalogModel {
  readonly id: string;
  readonly name: string;
  readonly category: 'instruct' | 'reference';
  readonly architecture: string;
  readonly quantization: string;
  readonly sizeLabel: string;
  readonly approximateBytes: number;
  readonly url: string;
  readonly bundled: boolean;
  readonly recommended?: boolean;
  readonly description: string;
  readonly vtensorReference: string;
}

export const MODEL_CATALOG: readonly CatalogModel[] = [
  {
    id: 'smollm2-135m-instruct-q4_0',
    name: 'SmolLM2 135M Instruct (Q4_0)',
    category: 'instruct',
    architecture: 'LLaMA (30 layers · 9/3 GQA · 576d)',
    quantization: 'Q4_0 / Q4_1 / Q8_0',
    sizeLabel: '87.6 MB',
    approximateBytes: 91_828_224,
    url: 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_0.gguf',
    bundled: false,
    recommended: true,
    description:
      'Fastest practical instruction-tuned assistant for browser execution. Uses ChatML templates, GPT-2 BPE tokenization, and Q4_0/Q4_1/Q8_0 Verdant kernels.',
    vtensorReference: 'HuggingFaceTB / bartowski · ChatML Instruct',
  },
  {
    id: 'smollm2-135m-instruct-q4_k_m',
    name: 'SmolLM2 135M Instruct (Q4_K_M)',
    category: 'instruct',
    architecture: 'LLaMA (30 layers · 9/3 GQA · 576d)',
    quantization: 'Q4_K / Q5_0 / Q6_K',
    sizeLabel: '100.6 MB',
    approximateBytes: 105_454_432,
    url: 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf',
    bundled: false,
    description:
      'Higher-precision mixed K-quant build of SmolLM2-135M-Instruct exercising Verdant Q4_K, Q5_0, Q6_K, and Q8_0 kernels.',
    vtensorReference: 'K-quant mixed precision · ChatML Instruct',
  },
  {
    id: 'smollm2-360m-instruct-q4_k_m',
    name: 'SmolLM2 360M Instruct (Q4_K_M)',
    category: 'instruct',
    architecture: 'LLaMA (32 layers · 15/5 GQA · 960d)',
    quantization: 'Q4_K / Q5_0 / Q6_K',
    sizeLabel: '258.1 MB',
    approximateBytes: 270_590_880,
    url: 'https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf',
    bundled: false,
    description:
      'Stronger reasoning, summarization, and coding accuracy in a compact 360M instruction-tuned LLaMA checkpoint.',
    vtensorReference: 'SmolLM2 360M · ChatML Instruct',
  },
  {
    id: 'qwen2.5-0.5b-instruct-q4_0',
    name: 'Qwen2.5 0.5B Instruct (Q4_0)',
    category: 'instruct',
    architecture: 'Qwen2 (24 layers · 14/2 GQA · 896d)',
    quantization: 'Q4_0 / Q8_0',
    sizeLabel: '408.9 MB',
    approximateBytes: 428_730_208,
    url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf',
    bundled: false,
    description:
      'Multilingual 500M assistant running the Qwen2 architecture (NeoX RoPE + Q/K/V attention biases + 151K BPE vocabulary) entirely on vTensor.',
    vtensorReference: 'Qwen2.5 · NeoX RoPE + QKV Biases',
  },
  {
    id: 'llama-3.2-1b-instruct-q4_k_m',
    name: 'Llama 3.2 1B Instruct (Q4_K_M)',
    category: 'instruct',
    architecture: 'LLaMA-3 (16 layers · 32/8 GQA · 2048d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '770.3 MB',
    approximateBytes: 807_694_464,
    url: 'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    bundled: false,
    description:
      '1B-parameter Llama 3.2 assistant streamed directly into OPFS vRAM with 3-tier KV scaling and Relaxed-SIMD VNNI Q4_K/Q6_K kernels.',
    vtensorReference: 'Meta Llama 3.2 1B · OPFS-Backed vRAM',
  },
  {
    id: 'qwen2.5-1.5b-instruct-q4_k_m',
    name: 'Qwen2.5 1.5B Instruct (Q4_K_M)',
    category: 'instruct',
    architecture: 'Qwen2 (28 layers · 12/2 GQA · 1536d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '1.04 GB',
    approximateBytes: 1_117_320_736,
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    bundled: false,
    description:
      '1.5B-parameter Qwen2.5 assistant with 152K vocabulary, NeoX RoPE, and QKV biases. Streams in 4 MiB chunks into OPFS without full-file JS heap copies.',
    vtensorReference: 'Qwen2.5 1.5B · OPFS-Backed vRAM',
  },
  {
    id: 'llama-3.2-3b-instruct-q4_k_m',
    name: 'Llama 3.2 3B Instruct (Q4_K_M)',
    category: 'instruct',
    architecture: 'LLaMA-3 (28 layers · 24/8 GQA · 3072d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '1.88 GB',
    approximateBytes: 2_019_377_696,
    url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    bundled: false,
    description:
      '3B-parameter Llama 3.2 model exceeding browser linear RAM ceilings, executed inside bounded WASM memory via OPFS-backed vRAM DIMM weight paging and striped LM-head streaming.',
    vtensorReference: 'Meta Llama 3.2 3B · OPFS vRAM Paged DIMM',
  },
  {
    id: 'stories260k-f32',
    name: 'TinyLLaMA Stories 260K',
    category: 'reference',
    architecture: 'LLaMA (5 layers · 8/4 GQA · 64d)',
    quantization: 'F32',
    sizeLabel: '1.13 MB',
    approximateBytes: 1_189_600,
    url: '/models/stories260K.gguf',
    bundled: true,
    description:
      'Bundled locally for instant zero-network boot and rapid graph verification. Generates short story continuations at 150+ tok/s.',
    vtensorReference: 'Bundled offline GGUF v3 · Instant boot',
  },
  {
    id: 'stories15m-q4_0',
    name: 'TinyLLaMA Stories 15M (Q4_0)',
    category: 'reference',
    architecture: 'LLaMA (6 layers · 6 heads · 288d)',
    quantization: 'Q4_0',
    sizeLabel: '18.19 MB',
    approximateBytes: 19_072_768,
    url: 'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf',
    bundled: false,
    description:
      'Exact quantized reference model validated in afflom/vTensor (VT-54 packed Q4_0 MUL_MAT and VT-58 6-layer FLASH_ATTN_EXT).',
    vtensorReference: 'afflom/vTensor VT-54 / VT-58 reference',
  },
];

export const RECOMMENDED_INSTRUCT_MODEL = MODEL_CATALOG[0];
export const BUNDLED_DEFAULT_MODEL = MODEL_CATALOG.find((m) => m.bundled)!;

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
