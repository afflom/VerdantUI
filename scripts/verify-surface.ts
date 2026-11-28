#!/usr/bin/env node
/**
 * Mechanical MCP Agent Surface Verifier (`scripts/verify-surface.ts`)
 *
 * Parity with `afflom/verdant-studio/.github/scripts/surface-check.py`:
 * Validates `surface/mcp-tools.json` against `MCP-SCHEMA-2026-07-28` structure,
 * uniqueness, JSON Schema 2020-12 inputSchema rules, and alignment with the
 * live `window.__VERDANT_OBSERVER__` API in `src/engine/verdantRootPrism.ts`.
 */

import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

interface McpTool {
  readonly name?: unknown;
  readonly description?: unknown;
  readonly inputSchema?: {
    readonly type?: unknown;
    readonly properties?: Record<string, unknown>;
    readonly required?: readonly unknown[];
    readonly additionalProperties?: unknown;
  };
  readonly annotations?: Record<string, unknown>;
}

interface McpToolsDoc {
  readonly schema?: unknown;
  readonly server?: { readonly name?: unknown; readonly version?: unknown; readonly authority?: unknown };
  readonly tools?: readonly McpTool[];
}

export function verifyMcpSurface(cwd = process.cwd()): { readonly ok: boolean; readonly toolCount: number; readonly errors: readonly string[] } {
  const root = resolve(cwd);
  const errors: string[] = [];
  const doc = JSON.parse(readFileSync(join(root, 'surface/mcp-tools.json'), 'utf8')) as McpToolsDoc;

  if (doc.schema !== 'MCP-SCHEMA-2026-07-28') {
    errors.push(`Expected schema "MCP-SCHEMA-2026-07-28", found ${JSON.stringify(doc.schema)}`);
  }
  if (!doc.server || typeof doc.server.name !== 'string' || !doc.server.name) {
    errors.push('surface/mcp-tools.json missing server.name');
  }
  if (!Array.isArray(doc.tools) || doc.tools.length < 5) {
    errors.push('surface/mcp-tools.json must publish at least 5 tools');
  }

  const seen = new Set<string>();
  const validName = /^[a-z][a-z0-9_]{2,63}$/;
  for (const tool of doc.tools ?? []) {
    if (typeof tool.name !== 'string' || !validName.test(tool.name)) {
      errors.push(`Invalid MCP tool name: ${JSON.stringify(tool.name)}`);
      continue;
    }
    if (seen.has(tool.name)) {
      errors.push(`Duplicate MCP tool name: ${tool.name}`);
    }
    seen.add(tool.name);
    if (typeof tool.description !== 'string' || tool.description.trim().length < 16) {
      errors.push(`Tool ${tool.name} needs a clear description (>= 16 chars)`);
    }
    if (!tool.inputSchema || tool.inputSchema.type !== 'object') {
      errors.push(`Tool ${tool.name} inputSchema.type must be "object"`);
    } else {
      const props = tool.inputSchema.properties ?? {};
      for (const reqKey of tool.inputSchema.required ?? []) {
        if (typeof reqKey !== 'string' || !(reqKey in props)) {
          errors.push(`Tool ${tool.name} required field ${JSON.stringify(reqKey)} not declared in properties`);
        }
      }
      if (tool.inputSchema.additionalProperties !== false) {
        errors.push(`Tool ${tool.name} inputSchema must set additionalProperties: false (closed surface)`);
      }
    }
  }

  // Verify live observer hook parity in verdantRootPrism.ts
  const rootPrismSrc = readFileSync(join(root, 'src/engine/verdantRootPrism.ts'), 'utf8');
  for (const symbol of ['perceive:', 'diagnose:', 'enact:', 'otlpJson:', 'holoDispatch:']) {
    if (!rootPrismSrc.includes(symbol)) {
      errors.push(`src/engine/verdantRootPrism.ts missing observer surface symbol "${symbol}"`);
    }
  }

  return { ok: errors.length === 0, toolCount: seen.size, errors };
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  const res = verifyMcpSurface(process.cwd());
  if (!res.ok) {
    console.error(`verify-surface: REFUSED (${res.errors.length} error(s))`);
    for (const e of res.errors) console.error(`  - ${e}`);
    process.exit(1);
  }
  console.log(`verify-surface: PASS · schema=MCP-SCHEMA-2026-07-28 · tools=${res.toolCount}`);
}
