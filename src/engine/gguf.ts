/**
 * GGUF v2/v3 binary format parser and GGML storage type descriptors.
 * Matches upstream ggml.h storage catalogs pinned by afflom/vTensor (VT-05/VT-06/VT-07).
 */

export enum GgmlType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
}

export interface GgmlTypeTrait {
  readonly type: GgmlType;
  readonly name: string;
  readonly blockSize: number;
  readonly typeSizeBytes: number;
  readonly quantized: boolean;
}

export const GGML_TYPE_TRAITS: Record<number, GgmlTypeTrait> = {
  [GgmlType.F32]: { type: GgmlType.F32, name: 'F32', blockSize: 1, typeSizeBytes: 4, quantized: false },
  [GgmlType.F16]: { type: GgmlType.F16, name: 'F16', blockSize: 1, typeSizeBytes: 2, quantized: false },
  [GgmlType.BF16]: { type: GgmlType.BF16, name: 'BF16', blockSize: 1, typeSizeBytes: 2, quantized: false },
  [GgmlType.Q4_0]: { type: GgmlType.Q4_0, name: 'Q4_0', blockSize: 32, typeSizeBytes: 18, quantized: true },
  [GgmlType.Q4_1]: { type: GgmlType.Q4_1, name: 'Q4_1', blockSize: 32, typeSizeBytes: 20, quantized: true },
  [GgmlType.Q5_0]: { type: GgmlType.Q5_0, name: 'Q5_0', blockSize: 32, typeSizeBytes: 22, quantized: true },
  [GgmlType.Q5_1]: { type: GgmlType.Q5_1, name: 'Q5_1', blockSize: 32, typeSizeBytes: 24, quantized: true },
  [GgmlType.Q8_0]: { type: GgmlType.Q8_0, name: 'Q8_0', blockSize: 32, typeSizeBytes: 34, quantized: true },
  [GgmlType.Q8_1]: { type: GgmlType.Q8_1, name: 'Q8_1', blockSize: 32, typeSizeBytes: 36, quantized: true },
  [GgmlType.Q2_K]: { type: GgmlType.Q2_K, name: 'Q2_K', blockSize: 256, typeSizeBytes: 84, quantized: true },
  [GgmlType.Q3_K]: { type: GgmlType.Q3_K, name: 'Q3_K', blockSize: 256, typeSizeBytes: 110, quantized: true },
  [GgmlType.Q4_K]: { type: GgmlType.Q4_K, name: 'Q4_K', blockSize: 256, typeSizeBytes: 144, quantized: true },
  [GgmlType.Q5_K]: { type: GgmlType.Q5_K, name: 'Q5_K', blockSize: 256, typeSizeBytes: 176, quantized: true },
  [GgmlType.Q6_K]: { type: GgmlType.Q6_K, name: 'Q6_K', blockSize: 256, typeSizeBytes: 210, quantized: true },
  [GgmlType.I8]: { type: GgmlType.I8, name: 'I8', blockSize: 1, typeSizeBytes: 1, quantized: false },
  [GgmlType.I16]: { type: GgmlType.I16, name: 'I16', blockSize: 1, typeSizeBytes: 2, quantized: false },
  [GgmlType.I32]: { type: GgmlType.I32, name: 'I32', blockSize: 1, typeSizeBytes: 4, quantized: false },
  [GgmlType.I64]: { type: GgmlType.I64, name: 'I64', blockSize: 1, typeSizeBytes: 8, quantized: false },
  [GgmlType.F64]: { type: GgmlType.F64, name: 'F64', blockSize: 1, typeSizeBytes: 8, quantized: false },
};

export function getGgmlTypeTrait(type: number): GgmlTypeTrait {
  const trait = GGML_TYPE_TRAITS[type];
  if (!trait) {
    throw new Error(`Unsupported GGML tensor storage type id: ${type}`);
  }
  return trait;
}

export function ggmlTensorByteSize(type: GgmlType, dims: readonly number[]): number {
  const trait = getGgmlTypeTrait(type);
  const ne0 = dims[0] ?? 1;
  if (ne0 % trait.blockSize !== 0) {
    throw new Error(
      `Tensor innermost dimension ${ne0} is not divisible by ${trait.name} block size ${trait.blockSize}`,
    );
  }
  const rowBytes = (ne0 / trait.blockSize) * trait.typeSizeBytes;
  let rows = 1;
  for (let i = 1; i < dims.length; i++) {
    rows *= dims[i];
  }
  return rowBytes * rows;
}

export type GgufMetadataValue =
  | number
  | bigint
  | boolean
  | string
  | GgufMetadataValue[];

export interface GgufTensorInfo {
  readonly name: string;
  readonly dims: readonly number[];
  readonly type: GgmlType;
  readonly typeName: string;
  readonly offset: number;
  readonly byteSize: number;
}

export interface LlamaHyperParams {
  readonly architecture: string;
  readonly name: string;
  readonly vocabSize: number;
  readonly contextLength: number;
  readonly embeddingLength: number;
  readonly feedForwardLength: number;
  readonly headCount: number;
  readonly headCountKv: number;
  readonly headDim: number;
  readonly blockCount: number;
  readonly ropeDimensionCount: number;
  readonly ropeFreqBase: number;
  readonly ropeFreqScale: number;
  readonly ropeType: 'norm' | 'neox';
  readonly rmsNormEps: number;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly unkTokenId: number;
  readonly addBosToken: boolean;
  readonly tokenizerModel: string;
  readonly preTokenizer: string;
  readonly chatTemplate: string | null;
  readonly primaryQuantization: string;
  /** Number of routed experts per layer (0 for dense models, e.g. 128 for Qwen3-30B-A3B). */
  readonly expertCount?: number;
  /** Number of active experts selected per token (0 for dense models, e.g. 8 for Qwen3-30B-A3B). */
  readonly expertUsedCount?: number;
  /** Intermediate FFN width per routed expert (defaults to feedForwardLength). */
  readonly expertFeedForwardLength?: number;
  /** Number of always-active shared experts (0 when none). */
  readonly expertSharedCount?: number;
  /** Whether Top-K router softmax weights are renormalized to sum to 1 (true on Qwen3-MoE). */
  readonly expertWeightsNorm?: boolean;
}

export interface ParsedGgufModel {
  readonly version: number;
  readonly alignment: number;
  readonly dataOffset: number;
  readonly fileByteLength: number;
  readonly metadata: Readonly<Record<string, GgufMetadataValue>>;
  readonly tensors: readonly GgufTensorInfo[];
  readonly tensorMap: ReadonlyMap<string, GgufTensorInfo>;
  readonly hparams: LlamaHyperParams;
  readonly rawBytes: Uint8Array;
}

enum GgufValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

const textDecoder = new TextDecoder('utf-8');

function alignUp(offset: number, alignment: number): number {
  const rem = offset % alignment;
  return rem === 0 ? offset : offset + (alignment - rem);
}

export function parseGGUF(buffer: ArrayBuffer | Uint8Array): ParsedGgufModel {
  const rawBytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (rawBytes.byteLength < 24) {
    throw new Error('Invalid GGUF file: buffer shorter than 24-byte header.');
  }

  const magic = textDecoder.decode(rawBytes.subarray(0, 4));
  if (magic !== 'GGUF') {
    throw new Error(`Invalid GGUF magic header "${magic}". Expected "GGUF".`);
  }

  const view = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
  let offset = 4;

  const version = view.getUint32(offset, true);
  offset += 4;
  if (version < 2 || version > 3) {
    throw new Error(`Unsupported GGUF version ${version}. Supported versions: 2, 3.`);
  }

  const tensorCount = Number(view.getBigUint64(offset, true));
  offset += 8;
  const kvCount = Number(view.getBigUint64(offset, true));
  offset += 8;

  const readString = (): string => {
    const len = Number(view.getBigUint64(offset, true));
    offset += 8;
    if (offset + len > rawBytes.byteLength) {
      throw new Error('Unexpected EOF while reading GGUF string.');
    }
    const value = textDecoder.decode(rawBytes.subarray(offset, offset + len));
    offset += len;
    return value;
  };

  const readValue = (valueType: number): GgufMetadataValue => {
    switch (valueType) {
      case GgufValueType.UINT8:
        return rawBytes[offset++];
      case GgufValueType.INT8: {
        const v = view.getInt8(offset);
        offset += 1;
        return v;
      }
      case GgufValueType.UINT16: {
        const v = view.getUint16(offset, true);
        offset += 2;
        return v;
      }
      case GgufValueType.INT16: {
        const v = view.getInt16(offset, true);
        offset += 2;
        return v;
      }
      case GgufValueType.UINT32: {
        const v = view.getUint32(offset, true);
        offset += 4;
        return v;
      }
      case GgufValueType.INT32: {
        const v = view.getInt32(offset, true);
        offset += 4;
        return v;
      }
      case GgufValueType.FLOAT32: {
        const v = view.getFloat32(offset, true);
        offset += 4;
        return v;
      }
      case GgufValueType.BOOL:
        return rawBytes[offset++] !== 0;
      case GgufValueType.STRING:
        return readString();
      case GgufValueType.ARRAY: {
        const elemType = view.getUint32(offset, true);
        offset += 4;
        const len = Number(view.getBigUint64(offset, true));
        offset += 8;
        const arr = new Array<GgufMetadataValue>(len);
        for (let i = 0; i < len; i++) {
          arr[i] = readValue(elemType);
        }
        return arr;
      }
      case GgufValueType.UINT64: {
        const v = view.getBigUint64(offset, true);
        offset += 8;
        return Number(v);
      }
      case GgufValueType.INT64: {
        const v = view.getBigInt64(offset, true);
        offset += 8;
        return Number(v);
      }
      case GgufValueType.FLOAT64: {
        const v = view.getFloat64(offset, true);
        offset += 8;
        return v;
      }
      default:
        throw new Error(`Unsupported GGUF metadata value type: ${valueType}`);
    }
  };

  const metadata: Record<string, GgufMetadataValue> = {};
  for (let i = 0; i < kvCount; i++) {
    const key = readString();
    const valueType = view.getUint32(offset, true);
    offset += 4;
    metadata[key] = readValue(valueType);
  }

  const tensors: GgufTensorInfo[] = [];
  const tensorMap = new Map<string, GgufTensorInfo>();

  for (let i = 0; i < tensorCount; i++) {
    const name = readString();
    const nDims = view.getUint32(offset, true);
    offset += 4;
    const dims: number[] = [];
    for (let d = 0; d < nDims; d++) {
      dims.push(Number(view.getBigUint64(offset, true)));
      offset += 8;
    }
    const type = view.getUint32(offset, true) as GgmlType;
    offset += 4;
    const tensorOffset = Number(view.getBigUint64(offset, true));
    offset += 8;

    const trait = getGgmlTypeTrait(type);
    const byteSize = ggmlTensorByteSize(type, dims);
    const info: GgufTensorInfo = {
      name,
      dims,
      type,
      typeName: trait.name,
      offset: tensorOffset,
      byteSize,
    };
    tensors.push(info);
    tensorMap.set(name, info);
  }

  const alignment = typeof metadata['general.alignment'] === 'number'
    ? (metadata['general.alignment'] as number)
    : 32;
  const dataOffset = alignUp(offset, alignment);

  const arch = typeof metadata['general.architecture'] === 'string'
    ? (metadata['general.architecture'] as string)
    : 'llama';

  const getNum = (key: string, fallback: number): number => {
    const val = metadata[key];
    return typeof val === 'number' && Number.isFinite(val) ? val : fallback;
  };

  const tokens = metadata['tokenizer.ggml.tokens'];
  const embdTensor = tensorMap.get('token_embd.weight');
  const vocabSize = Array.isArray(tokens)
    ? tokens.length
    : (embdTensor?.dims[1] ?? 32000);

  const embeddingLength = getNum(`${arch}.embedding_length`, embdTensor?.dims[0] ?? 64);
  const headCount = getNum(`${arch}.attention.head_count`, 8);
  const headCountKv = getNum(`${arch}.attention.head_count_kv`, headCount);
  const headDim = getNum(`${arch}.attention.key_length`, Math.floor(embeddingLength / headCount));

  const primaryWeight = tensorMap.get('blk.0.attn_q.weight') ?? embdTensor;
  const primaryQuantization = primaryWeight ? primaryWeight.typeName : 'F32';

  const tokenizerModel =
    typeof metadata['tokenizer.ggml.model'] === 'string'
      ? (metadata['tokenizer.ggml.model'] as string)
      : 'llama';

  const addBosMeta = metadata['tokenizer.ggml.add_bos_token'];
  const addBosToken =
    typeof addBosMeta === 'boolean' ? addBosMeta : tokenizerModel !== 'gpt2';

  const preTokenizer =
    typeof metadata['tokenizer.ggml.pre'] === 'string'
      ? (metadata['tokenizer.ggml.pre'] as string)
      : tokenizerModel === 'gpt2'
        ? 'gpt2'
        : 'default';

  const chatTemplate =
    typeof metadata['tokenizer.chat_template'] === 'string' &&
    (metadata['tokenizer.chat_template'] as string).trim().length > 0
      ? (metadata['tokenizer.chat_template'] as string)
      : null;

  const NEOX_ROPE_ARCHS = new Set([
    'qwen2',
    'qwen2moe',
    'qwen3',
    'qwen3moe',
    'phi3',
    'gemma',
    'gemma2',
    'gemma3',
    'stablelm',
    'starcoder2',
    'olmo2',
    'falcon',
    'deepseek2',
  ]);
  const ropeType: 'norm' | 'neox' = NEOX_ROPE_ARCHS.has(arch) ? 'neox' : 'norm';
  const ropeScalingFactor = getNum(`${arch}.rope.scaling.factor`, 1.0);
  const ropeFreqScale =
    ropeScalingFactor > 0 && Number.isFinite(ropeScalingFactor) ? 1.0 / ropeScalingFactor : 1.0;

  const feedForwardLength = getNum(`${arch}.feed_forward_length`, embeddingLength * 4);
  const expertCount = getNum(`${arch}.expert_count`, 0);
  const expertUsedCount = getNum(`${arch}.expert_used_count`, expertCount > 0 ? Math.min(2, expertCount) : 0);
  const expertFeedForwardLength = getNum(`${arch}.expert_feed_forward_length`, feedForwardLength);
  const expertSharedCount = getNum(`${arch}.expert_shared_count`, 0);
  const expertWeightsNormMeta = metadata[`${arch}.expert_weights_norm`];
  const expertWeightsNorm =
    typeof expertWeightsNormMeta === 'boolean'
      ? expertWeightsNormMeta
      : arch === 'qwen3moe' || arch === 'mixtral';

  const hparams: LlamaHyperParams = {
    architecture: arch,
    name: typeof metadata['general.name'] === 'string' ? (metadata['general.name'] as string) : arch,
    vocabSize,
    contextLength: getNum(`${arch}.context_length`, 2048),
    embeddingLength,
    feedForwardLength,
    headCount,
    headCountKv,
    headDim,
    blockCount: getNum(`${arch}.block_count`, 1),
    ropeDimensionCount: getNum(`${arch}.rope.dimension_count`, headDim),
    ropeFreqBase: getNum(`${arch}.rope.freq_base`, 10000.0),
    ropeFreqScale,
    ropeType,
    rmsNormEps: getNum(`${arch}.attention.layer_norm_rms_epsilon`, 1e-5),
    bosTokenId: getNum('tokenizer.ggml.bos_token_id', 1),
    eosTokenId: getNum('tokenizer.ggml.eos_token_id', 2),
    unkTokenId: getNum('tokenizer.ggml.unknown_token_id', 0),
    addBosToken,
    tokenizerModel,
    preTokenizer,
    chatTemplate,
    primaryQuantization,
    expertCount,
    expertUsedCount,
    expertFeedForwardLength,
    expertSharedCount,
    expertWeightsNorm,
  };

  return {
    version,
    alignment,
    dataOffset,
    fileByteLength: rawBytes.byteLength,
    metadata,
    tensors,
    tensorMap,
    hparams,
    rawBytes,
  };
}

/**
 * Header-only streaming GGUF parser backed by an OPFS or memory `VramBackingStore`.
 * Reads ONLY the GGUF header + KV metadata + tensor directory (~2–12 MiB even for 3B–7B models),
 * never loading gigabytes of tensor weights into JS heap memory.
 */
export function parseGgufFromBacking(backing: {
  readonly byteLength: number;
  readRange(fileOffset: number, dst: Uint8Array, dstOffset?: number, length?: number): number;
}): ParsedGgufModel {
  const totalBytes = backing.byteLength;
  if (totalBytes < 24) {
    throw new Error('Invalid GGUF backing: shorter than 24-byte header.');
  }

  // Size initial header read window to parse large 150K-vocab Qwen3/Llama3 GGUF headers in 1 pass
  // instead of wasting 2-3x redundant parse retries (supports up to 128 MiB headers).
  const initialWindow = totalBytes >= 500 * 1024 * 1024 ? 12 * 1024 * 1024 : 4 * 1024 * 1024;
  let windowBytes = Math.min(totalBytes, initialWindow);
  const maxHeaderCap = Math.min(totalBytes, 128 * 1024 * 1024);

  while (true) {
    const headerBuf = new Uint8Array(windowBytes);
    const read = backing.readRange(0, headerBuf, 0, windowBytes);
    if (read < 24) {
      throw new Error('Failed to read GGUF header from vRAM backing store.');
    }
    try {
      const parsed = parseGGUF(read === windowBytes ? headerBuf : headerBuf.subarray(0, read));
      // Verify every tensor's byte extent fits inside the backing file (catches truncated GGUFs early!)
      for (let i = 0; i < parsed.tensors.length; i++) {
        const t = parsed.tensors[i];
        const end = parsed.dataOffset + t.offset + t.byteSize;
        if (end > totalBytes) {
          throw new Error(
            `Truncated GGUF backing: tensor "${t.name}" requires byte ${end}, but backing size is ${totalBytes}.`,
          );
        }
      }
      return {
        ...parsed,
        fileByteLength: totalBytes,
        // Drop reference to header buffer so only parsed metadata + tensor directory stay in JS heap
        rawBytes: new Uint8Array(0),
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isTruncated =
        !msg.startsWith('Truncated GGUF backing:') &&
        (err instanceof RangeError ||
          msg.includes('Unexpected EOF') ||
          msg.includes('Offset is outside the bounds'));
      if (isTruncated && windowBytes < maxHeaderCap) {
        windowBytes = Math.min(maxHeaderCap, windowBytes * 2);
        continue;
      }
      throw err;
    }
  }
}

/**
 * Async header-only GGUF parser for demand-paged remote backings (`opfs-demand`).
 * Prefetches GGUF header slabs in parallel via `backing.prefetchRanges` with `AbortSignal`
 * before parsing, avoiding blocking synchronous XHR stalls on large 150K-vocab headers.
 */
export async function parseGgufFromBackingAsync(
  backing: {
    readonly byteLength: number;
    readRange(fileOffset: number, dst: Uint8Array, dstOffset?: number, length?: number): number;
    prefetchRanges?(
      ranges: readonly { readonly fileOffset: number; readonly byteLength: number }[],
      options?: {
        readonly signal?: AbortSignal;
        readonly concurrency?: number;
        readonly onProgress?: (doneSlabs: number, totalSlabs: number, bytesFetched: number) => void;
      },
    ): Promise<void>;
    readRangesDirectAsync?(
      ranges: readonly { readonly fileOffset: number; readonly byteLength: number }[],
      options?: {
        readonly signal?: AbortSignal;
        readonly concurrency?: number;
        readonly maxGapBytes?: number;
        readonly onProgress?: (doneGroups: number, totalGroups: number) => void;
      },
    ): Promise<Uint8Array[]>;
  },
  options?: {
    readonly signal?: AbortSignal;
    readonly onProgress?: (detail: string) => void;
  },
): Promise<ParsedGgufModel> {
  if (!backing.readRangesDirectAsync && !backing.prefetchRanges) {
    return parseGgufFromBacking(backing);
  }
  const totalBytes = backing.byteLength;
  if (totalBytes < 24) {
    throw new Error('Invalid GGUF backing: shorter than 24-byte header.');
  }

  const initialAsyncWindow =
    totalBytes >= 500 * 1024 * 1024 ? 12 * 1024 * 1024 : 2 * 1024 * 1024;
  let windowBytes = Math.min(totalBytes, initialAsyncWindow);
  const maxHeaderCap = Math.min(totalBytes, 128 * 1024 * 1024);
  let loadedBuf = new Uint8Array(0);

  while (true) {
    if (options?.signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
    if (windowBytes > loadedBuf.byteLength) {
      options?.onProgress?.(
        `Reading model header (${(windowBytes / (1024 * 1024)).toFixed(1)} MB)`,
      );
      const deltaOffset = loadedBuf.byteLength;
      const deltaLen = windowBytes - deltaOffset;
      if (backing.readRangesDirectAsync) {
        const [chunk] = await backing.readRangesDirectAsync(
          [{ fileOffset: deltaOffset, byteLength: deltaLen }],
          { signal: options?.signal },
        );
        const nextBuf = new Uint8Array(deltaOffset + chunk.byteLength);
        nextBuf.set(loadedBuf, 0);
        nextBuf.set(chunk, deltaOffset);
        loadedBuf = nextBuf;
      } else {
        await backing.prefetchRanges!(
          [{ fileOffset: deltaOffset, byteLength: deltaLen }],
          { signal: options?.signal },
        );
        const nextBuf = new Uint8Array(windowBytes);
        const read = backing.readRange(0, nextBuf, 0, windowBytes);
        loadedBuf = read === windowBytes ? nextBuf : nextBuf.subarray(0, read);
      }
    }

    if (loadedBuf.byteLength < 24) {
      throw new Error('Failed to read GGUF header from vRAM backing store.');
    }
    try {
      const parsed = parseGGUF(loadedBuf);
      for (let i = 0; i < parsed.tensors.length; i++) {
        const t = parsed.tensors[i];
        const end = parsed.dataOffset + t.offset + t.byteSize;
        if (end > totalBytes) {
          throw new Error(
            `Truncated GGUF backing: tensor "${t.name}" requires byte ${end}, but backing size is ${totalBytes}.`,
          );
        }
      }
      return {
        ...parsed,
        fileByteLength: totalBytes,
        rawBytes: new Uint8Array(0),
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isTruncated =
        !msg.startsWith('Truncated GGUF backing:') &&
        (err instanceof RangeError ||
          msg.includes('Unexpected EOF') ||
          msg.includes('Offset is outside the bounds'));
      if (isTruncated && windowBytes < maxHeaderCap) {
        windowBytes = Math.min(maxHeaderCap, windowBytes * 2);
        continue;
      }
      throw err;
    }
  }
}

export function sliceGgufTensorBytes(model: ParsedGgufModel, info: GgufTensorInfo): Uint8Array {
  const start = model.dataOffset + info.offset;
  const end = start + info.byteSize;
  if (end > model.rawBytes.byteLength) {
    throw new Error(
      `Tensor "${info.name}" exceeds GGUF file bounds (${end} > ${model.rawBytes.byteLength}).`,
    );
  }
  return model.rawBytes.subarray(start, end);
}

