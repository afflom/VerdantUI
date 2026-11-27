/**
 * GGUF Model Catalog for VerdantUI (0.5B to 30B–32B).
 *
 * Pure factual model metadata (verified Hugging Face GGUF URLs and exact transformer geometry)
 * without fabricated harnesses or hardcoded system prompts.
 *
 * Development/conformance checkpoints (`stories260K.gguf`) are excluded from `MODEL_CATALOG` and
 * exported solely as `INTERNAL_BOOTSTRAP_MODEL` for silent zero-network initialization.
 */

import { GgmlType } from './gguf';

export type ModelCategory = 'all' | 'coding' | 'general' | 'compact';

export interface ModelCategoryInfo {
  readonly id: ModelCategory;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
}

export const MODEL_CATEGORIES: readonly ModelCategoryInfo[] = [
  {
    id: 'all',
    label: 'All Models',
    shortLabel: 'All',
    description: 'Verified open-weight GGUF checkpoints from 0.5B to 32B.',
  },
  {
    id: 'coding',
    label: 'Coding',
    shortLabel: 'Coding',
    description: 'Code-specialized checkpoints from 7B to 32B (dense & Sparse MoE).',
  },
  {
    id: 'general',
    label: 'General',
    shortLabel: 'General',
    description: 'General-purpose instruction and reasoning checkpoints from 3B to 30B.',
  },
  {
    id: 'compact',
    label: 'Compact',
    shortLabel: 'Compact',
    description: 'Lightweight 0.5B–1.5B checkpoints for low-latency local inference.',
  },
];

export interface CatalogModelGeometry {
  readonly architecture: string;
  readonly blockCount: number;
  readonly embeddingLength: number;
  readonly feedForwardLength: number;
  readonly headCount: number;
  readonly headCountKv: number;
  readonly headDim: number;
  readonly vocabSize: number;
  readonly nativeContextTokens: number;
  readonly dominantGgmlType: GgmlType;
  readonly paramsBillions: number;
  readonly activeParamsBillions: number;
  readonly isMoe: boolean;
  readonly expertCount: number;
  readonly expertUsedCount: number;
  readonly expertFeedForwardLength: number;
}

export interface CatalogModel {
  readonly id: string;
  readonly name: string;
  readonly shortName: string;
  readonly category: Exclude<ModelCategory, 'all'>;
  readonly architecture: string;
  readonly quantization: string;
  readonly sizeLabel: string;
  readonly approximateBytes: number;
  readonly url: string;
  readonly bundled: boolean;
  readonly description: string;
  readonly geometry: CatalogModelGeometry;
}

/**
 * Production user-facing GGUF catalog — factual model metadata from 0.5B up to 30B–32B.
 * Conformance/dev models (`stories260K`, `stories15M`) are intentionally omitted here.
 */
export const MODEL_CATALOG: readonly CatalogModel[] = [
  {
    id: 'qwen3-coder-30b-a3b-q4_k_m',
    name: 'Qwen3-Coder 30B-A3B Instruct (Q4_K_M)',
    shortName: 'Qwen3-Coder 30B-A3B',
    category: 'coding',
    architecture: 'Qwen3-MoE (48L · 128E / 8 active · 3.3B active)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '18.56 GB',
    approximateBytes: 18_556_689_568,
    url: 'https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf',
    bundled: false,
    description:
      '30.5B Mixture-of-Experts coding model activating 3.3B parameters per token with 256K native context.',
    geometry: {
      architecture: 'qwen3moe',
      blockCount: 48,
      embeddingLength: 2048,
      feedForwardLength: 768,
      headCount: 32,
      headCountKv: 4,
      headDim: 128,
      vocabSize: 151_936,
      nativeContextTokens: 262_144,
      dominantGgmlType: GgmlType.Q4_K,
      paramsBillions: 30.5,
      activeParamsBillions: 3.3,
      isMoe: true,
      expertCount: 128,
      expertUsedCount: 8,
      expertFeedForwardLength: 768,
    },
  },
  {
    id: 'qwen3-coder-30b-a3b-q2_k',
    name: 'Qwen3-Coder 30B-A3B Instruct (Q2_K)',
    shortName: 'Qwen3-Coder 30B-A3B Q2_K',
    category: 'coding',
    architecture: 'Qwen3-MoE (48L · 128E / 8 active · 3.3B active)',
    quantization: 'Q2_K / Q4_K',
    sizeLabel: '11.26 GB',
    approximateBytes: 11_258_612_896,
    url: 'https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf',
    bundled: false,
    description:
      'Compact 2-bit quantized build of Qwen3-Coder 30B-A3B reducing file size by 39% while keeping 256K context.',
    geometry: {
      architecture: 'qwen3moe',
      blockCount: 48,
      embeddingLength: 2048,
      feedForwardLength: 768,
      headCount: 32,
      headCountKv: 4,
      headDim: 128,
      vocabSize: 151_936,
      nativeContextTokens: 262_144,
      dominantGgmlType: GgmlType.Q2_K,
      paramsBillions: 30.5,
      activeParamsBillions: 3.3,
      isMoe: true,
      expertCount: 128,
      expertUsedCount: 8,
      expertFeedForwardLength: 768,
    },
  },
  {
    id: 'qwen2.5-coder-32b-instruct-q4_0',
    name: 'Qwen2.5-Coder 32B Instruct (Q4_0)',
    shortName: 'Qwen2.5-Coder 32B',
    category: 'coding',
    architecture: 'Qwen2 Dense (64L · 40/8 GQA · 5120d)',
    quantization: 'Q4_0 / Q8_0',
    sizeLabel: '18.71 GB',
    approximateBytes: 18_711_010_272,
    url: 'https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q4_0.gguf',
    bundled: false,
    description:
      '32B dense code model for multi-file software engineering, code review, and refactoring.',
    geometry: {
      architecture: 'qwen2',
      blockCount: 64,
      embeddingLength: 5120,
      feedForwardLength: 27_648,
      headCount: 40,
      headCountKv: 8,
      headDim: 128,
      vocabSize: 152_064,
      nativeContextTokens: 32_768,
      dominantGgmlType: GgmlType.Q4_0,
      paramsBillions: 32.5,
      activeParamsBillions: 32.5,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 27_648,
    },
  },
  {
    id: 'qwen2.5-coder-14b-instruct-q4_0',
    name: 'Qwen2.5-Coder 14B Instruct (Q4_0)',
    shortName: 'Qwen2.5-Coder 14B',
    category: 'coding',
    architecture: 'Qwen2 Dense (48L · 40/8 GQA · 5120d)',
    quantization: 'Q4_0 / Q8_0',
    sizeLabel: '8.54 GB',
    approximateBytes: 8_544_268_512,
    url: 'https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_0.gguf',
    bundled: false,
    description:
      '14B dense code model balancing strong programming accuracy with a smaller download footprint.',
    geometry: {
      architecture: 'qwen2',
      blockCount: 48,
      embeddingLength: 5120,
      feedForwardLength: 13_824,
      headCount: 40,
      headCountKv: 8,
      headDim: 128,
      vocabSize: 152_064,
      nativeContextTokens: 32_768,
      dominantGgmlType: GgmlType.Q4_0,
      paramsBillions: 14.7,
      activeParamsBillions: 14.7,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 13_824,
    },
  },
  {
    id: 'qwen2.5-coder-7b-instruct-q4_0',
    name: 'Qwen2.5-Coder 7B Instruct (Q4_0)',
    shortName: 'Qwen2.5-Coder 7B',
    category: 'coding',
    architecture: 'Qwen2 Dense (28L · 28/4 GQA · 3584d)',
    quantization: 'Q4_0 / Q8_0',
    sizeLabel: '4.44 GB',
    approximateBytes: 4_444_121_888,
    url: 'https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_0.gguf',
    bundled: false,
    description:
      '7B dense code model suited for everyday code generation, debugging, and refactoring.',
    geometry: {
      architecture: 'qwen2',
      blockCount: 28,
      embeddingLength: 3584,
      feedForwardLength: 18_944,
      headCount: 28,
      headCountKv: 4,
      headDim: 128,
      vocabSize: 152_064,
      nativeContextTokens: 32_768,
      dominantGgmlType: GgmlType.Q4_0,
      paramsBillions: 7.6,
      activeParamsBillions: 7.6,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 18_944,
    },
  },
  {
    id: 'qwen3-30b-a3b-q4_k_m',
    name: 'Qwen3 30B-A3B Instruct (Q4_K_M)',
    shortName: 'Qwen3 30B-A3B',
    category: 'general',
    architecture: 'Qwen3-MoE (48L · 128E / 8 active · 3.3B active)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '18.56 GB',
    approximateBytes: 18_556_686_752,
    url: 'https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf',
    bundled: false,
    description:
      '30.5B Mixture-of-Experts general instruction model activating 3.3B parameters per token with 256K context.',
    geometry: {
      architecture: 'qwen3moe',
      blockCount: 48,
      embeddingLength: 2048,
      feedForwardLength: 768,
      headCount: 32,
      headCountKv: 4,
      headDim: 128,
      vocabSize: 151_936,
      nativeContextTokens: 262_144,
      dominantGgmlType: GgmlType.Q4_K,
      paramsBillions: 30.5,
      activeParamsBillions: 3.3,
      isMoe: true,
      expertCount: 128,
      expertUsedCount: 8,
      expertFeedForwardLength: 768,
    },
  },
  {
    id: 'qwen3-8b-q4_k_m',
    name: 'Qwen3 8B Instruct (Q4_K_M)',
    shortName: 'Qwen3 8B',
    category: 'general',
    architecture: 'Qwen3 Dense (36L · 32/8 GQA · 4096d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '5.03 GB',
    approximateBytes: 5_027_784_512,
    url: 'https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf',
    bundled: false,
    description:
      '8B dense reasoning and instruction model with 40K context support.',
    geometry: {
      architecture: 'qwen3',
      blockCount: 36,
      embeddingLength: 4096,
      feedForwardLength: 12_288,
      headCount: 32,
      headCountKv: 8,
      headDim: 128,
      vocabSize: 151_936,
      nativeContextTokens: 40_960,
      dominantGgmlType: GgmlType.Q4_K,
      paramsBillions: 8.2,
      activeParamsBillions: 8.2,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 12_288,
    },
  },
  {
    id: 'qwen3-4b-q4_k_m',
    name: 'Qwen3 4B Instruct (Q4_K_M)',
    shortName: 'Qwen3 4B',
    category: 'general',
    architecture: 'Qwen3 Dense (36L · 32/8 GQA · 2560d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '2.50 GB',
    approximateBytes: 2_497_281_312,
    url: 'https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf',
    bundled: false,
    description:
      '4B dense reasoning and instruction model with 40K context support.',
    geometry: {
      architecture: 'qwen3',
      blockCount: 36,
      embeddingLength: 2560,
      feedForwardLength: 9728,
      headCount: 32,
      headCountKv: 8,
      headDim: 128,
      vocabSize: 151_936,
      nativeContextTokens: 40_960,
      dominantGgmlType: GgmlType.Q4_K,
      paramsBillions: 4.0,
      activeParamsBillions: 4.0,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 9728,
    },
  },
  {
    id: 'llama-3.2-3b-instruct-q4_k_m',
    name: 'Llama 3.2 3B Instruct (Q4_K_M)',
    shortName: 'Llama 3.2 3B',
    category: 'general',
    architecture: 'LLaMA-3 (28L · 24/8 GQA · 3072d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '1.88 GB',
    approximateBytes: 2_019_377_696,
    url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    bundled: false,
    description:
      '3B instruction model with 128K context for fast everyday conversations and analysis.',
    geometry: {
      architecture: 'llama',
      blockCount: 28,
      embeddingLength: 3072,
      feedForwardLength: 8192,
      headCount: 24,
      headCountKv: 8,
      headDim: 128,
      vocabSize: 128_256,
      nativeContextTokens: 131_072,
      dominantGgmlType: GgmlType.Q4_K,
      paramsBillions: 3.2,
      activeParamsBillions: 3.2,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 8192,
    },
  },
  {
    id: 'qwen2.5-1.5b-instruct-q4_k_m',
    name: 'Qwen2.5 1.5B Instruct (Q4_K_M)',
    shortName: 'Qwen2.5 1.5B',
    category: 'compact',
    architecture: 'Qwen2 Dense (28L · 12/2 GQA · 1536d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '1.04 GB',
    approximateBytes: 1_117_320_736,
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    bundled: false,
    description:
      'Fast 1.5B instruct model for summarization and quick local drafting with a 1 GB footprint.',
    geometry: {
      architecture: 'qwen2',
      blockCount: 28,
      embeddingLength: 1536,
      feedForwardLength: 8960,
      headCount: 12,
      headCountKv: 2,
      headDim: 128,
      vocabSize: 151_936,
      nativeContextTokens: 32_768,
      dominantGgmlType: GgmlType.Q4_K,
      paramsBillions: 1.5,
      activeParamsBillions: 1.5,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 8960,
    },
  },
  {
    id: 'qwen2.5-0.5b-instruct-q4_k_m',
    name: 'Qwen2.5 0.5B Instruct (Q4_K_M)',
    shortName: 'Qwen2.5 0.5B',
    category: 'compact',
    architecture: 'Qwen2 Dense (24L · 14/2 GQA · 896d)',
    quantization: 'Q4_K / Q6_K',
    sizeLabel: '491.4 MB',
    approximateBytes: 491_400_032,
    url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf',
    bundled: false,
    description:
      'Lightweight 0.5B model for quick classification, short summaries, and low-memory devices.',
    geometry: {
      architecture: 'qwen2',
      blockCount: 24,
      embeddingLength: 896,
      feedForwardLength: 4864,
      headCount: 14,
      headCountKv: 2,
      headDim: 64,
      vocabSize: 151_936,
      nativeContextTokens: 32_768,
      dominantGgmlType: GgmlType.Q4_K,
      paramsBillions: 0.5,
      activeParamsBillions: 0.5,
      isMoe: false,
      expertCount: 0,
      expertUsedCount: 0,
      expertFeedForwardLength: 4864,
    },
  },
];

/**
 * Internal-only zero-network bootstrap checkpoint (`stories260K.gguf`).
 * Strictly excluded from `MODEL_CATALOG` and the user-facing Model Manager UI; used only so the
 * Web Worker initializes cleanly offline before the user selects a model.
 */
export const INTERNAL_BOOTSTRAP_MODEL: CatalogModel = {
  id: 'stories260k-f32',
  name: 'Bootstrap Runtime (Unbound)',
  shortName: 'Bootstrap',
  category: 'compact',
  architecture: 'LLaMA (5L · 64d)',
  quantization: 'F32',
  sizeLabel: '1.13 MB',
  approximateBytes: 1_189_600,
  url: 'models/stories260K.gguf',
  bundled: true,
  description: 'Internal zero-network bootstrap checkpoint.',
  geometry: {
    architecture: 'llama',
    blockCount: 5,
    embeddingLength: 64,
    feedForwardLength: 172,
    headCount: 8,
    headCountKv: 4,
    headDim: 8,
    vocabSize: 512,
    nativeContextTokens: 256,
    dominantGgmlType: GgmlType.F32,
    paramsBillions: 0.00026,
    activeParamsBillions: 0.00026,
    isMoe: false,
    expertCount: 0,
    expertUsedCount: 0,
    expertFeedForwardLength: 172,
  },
};

export interface MultimodalCoreModelSpec {
  readonly id: string;
  readonly modality: 'vision' | 'audio' | 'image-synthesis';
  readonly name: string;
  readonly approximateBytes: number;
  readonly url: string;
  readonly companionUrl?: string;
}

/**
 * Verified open-weight multimodal core models automatically managed on-device via OPFS
 * and executed on VerdantUI's shared vTensor WASM SIMD substrate (zero user model uploads).
 */
export const MULTIMODAL_CORE_MODELS: readonly MultimodalCoreModelSpec[] = [
  {
    id: 'smolvlm2-256m-video-mmproj-q8',
    modality: 'vision',
    name: 'SmolVLM2 256M Vision & Video Projector (Q8_0)',
    approximateBytes: 109_000_000,
    url: 'https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf',
    companionUrl:
      'https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/SmolVLM2-256M-Video-Instruct-Q8_0.gguf',
  },
  {
    id: 'whisper-tiny-en-q8',
    modality: 'audio',
    name: 'Whisper Tiny EN Speech Encoder-Decoder (Q8_0)',
    approximateBytes: 43_600_000,
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q8_0.bin',
  },
  {
    id: 'sdxs-512-q8-taesd',
    modality: 'image-synthesis',
    name: 'SDXS-512 1-Step Distilled Diffusion (Q8_0) + TAESD',
    approximateBytes: 640_000_000,
    url: 'https://huggingface.co/concedo/sdxs-512-tinySDdistilled-GGUF/resolve/3144d898d61492f8382ffcabec055733fc5b2a0e/sdxs-512-tinySDdistilled_Q8_0.gguf',
  },
];

export function isInternalBootstrapModel(modelId: string | null | undefined): boolean {
  return !modelId || modelId === INTERNAL_BOOTSTRAP_MODEL.id;
}

export function getModelsByCategory(category: ModelCategory): readonly CatalogModel[] {
  if (category === 'all') return MODEL_CATALOG;
  return MODEL_CATALOG.filter((m) => m.category === category);
}

export function findCatalogModelById(id: string): CatalogModel | undefined {
  return MODEL_CATALOG.find((m) => m.id === id);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
