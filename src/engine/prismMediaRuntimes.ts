/**
 * Procedural Media Runtimes Barrel (`src/engine/prismMediaRuntimes.ts`)
 *
 * Re-exports the modular visual, audio, and editorial/research/utility/space
 * procedural runtimes while keeping all existing imports intact.
 */

export { buildPictureRuntimeJs, buildVideoRuntimeJs } from './prismVisualRuntimes';
export { buildAudioRuntimeJs } from './prismAudioRuntimes';
export {
  buildWrittenWorkRuntimeJs,
  buildResearchProgramRuntimeJs,
  buildToolOrganRuntimeJs,
  buildSpaceRuntimeJs,
} from './prismEditorialRuntimes';
