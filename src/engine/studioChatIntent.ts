/**
 * Lightweight Natural-Language Verdant Studio Chat Intent Detector (`src/engine/studioChatIntent.ts`)
 *
 * Kept dependency-free so `cognitiveObserver.ts` detects Studio creation, refinement,
 * Space curation, and Gallery browsing intents in <0.1 ms without pulling procedural
 * media synthesizers into VerdantUI's initial boot chunk.
 */

import { GALLERY_LABEL_BY_TYPE, type GalleryMediaType } from './contentGalleryStore';

export interface DetectedStudioChatIntent {
  readonly toolName:
    | 'studio_create_artifact'
    | 'studio_refine_prism'
    | 'studio_manage_spaces'
    | 'studio_browse_gallery';
  readonly args: Record<string, unknown>;
  readonly confidenceQ10: number;
  readonly reason: string;
}

function extractGalleryTypeFromText(lower: string): GalleryMediaType | null {
  if (/\b(pictures?|images?|artworks?|photos?|illustrations?)\b/.test(lower)) return 'picture';
  if (/\b(videos?|animations?|clips?|movies?|reels?)\b/.test(lower)) return 'video';
  if (/\b(audio|music|songs?|tracks?|soundscapes?|sounds?)\b/.test(lower)) return 'audio';
  if (/\b(documents?|writing|stories|novels?|essays?|manuscripts?)\b/.test(lower)) return 'written-work';
  if (/\b(games?|puzzles?|arcade)\b/.test(lower)) return 'interactive-game';
  if (/\b(research|studies|experiments?)\b/.test(lower)) return 'research-program';
  if (/\b(utilities|tools?|habits?|schedules?|calculators?|workflows?)\b/.test(lower)) return 'tool-organ';
  return null;
}

/**
 * Detects unambiguous natural-language Verdant Studio creation, refinement, Space curation,
 * and Gallery management intents in chat threads while NEVER hijacking informational questions.
 */
export function detectStudioChatIntent(
  text: string,
  hasAttachment = false,
): DetectedStudioChatIntent | null {
  if (hasAttachment) return null;
  const trimmed = text.trim();
  if (!trimmed || trimmed.length > 420) return null;
  const lower = trimmed.toLowerCase();

  // Never hijack pure informational / explanatory questions
  if (/^(?:what|why|how|who|when|where|can\s+you\s+explain|explain|describe|define|compare|tell\s+me\s+about)\b/i.test(trimmed)) {
    return null;
  }

  // 1. Browse a MediaType Gallery or list Prisms / Spaces
  if (/^(?:show|open|view|list|browse)\s+(?:my\s+)?(?:all\s+)?(pictures?|images?|videos?|audio|music|documents?|games?|research|utilities|tools?)(?:\s+gallery)?[?.!]*$/i.test(trimmed)) {
    const mt = extractGalleryTypeFromText(lower);
    if (mt) {
      return {
        toolName: 'studio_browse_gallery',
        args: { mediaType: mt },
        confidenceQ10: 965,
        reason: `Open ${GALLERY_LABEL_BY_TYPE[mt]} gallery.`,
      };
    }
  }
  if (/^(?:list|show|view)\s+(?:all\s+)?(?:my\s+)?(?:prisms?|spaces?|creations?|apps?)[?.!]*$/i.test(trimmed)) {
    const spacesOnly = /\bspaces?\b/.test(lower);
    return {
      toolName: 'studio_manage_spaces',
      args: { action: 'list', spacesOnly },
      confidenceQ10: 965,
      reason: spacesOnly ? 'List all Spaces.' : 'List all Prisms and Spaces.',
    };
  }

  // 2. Create a Space (with optional inline curation)
  const createSpaceMatch = trimmed.match(
    /^(?:please\s+)?(?:create|make|start|new)\s+(?:a\s+)?(?:new\s+)?space\s+(?:called|named)\s+["'“]?([^"'”.,!?]{2,56})["'”]?(?:[\s,]+(?:and\s+)?(.+))?$/i,
  );
  if (createSpaceMatch) {
    const spaceName = createSpaceMatch[1]!.trim();
    const rest = (createSpaceMatch[2] ?? '').trim();
    return {
      toolName: 'studio_manage_spaces',
      args: {
        action: 'create',
        name: spaceName,
        description: rest || undefined,
        addLatest: /\badd\s+(?:that|this|the\s+last|it)\b/i.test(rest),
      },
      confidenceQ10: 970,
      reason: `Create Space Prism "${spaceName}".`,
    };
  }

  // 3. Add an artifact to a Space
  const addToSpaceMatch = trimmed.match(
    /^(?:please\s+)?(?:add|put|save|move)\s+(?:(?:that|this|it|the\s+(?:last|latest)\s+\w+)|["'“]?([^"'”]{2,48})["'”]?)\s+(?:to|into|in)\s+(?:(?:my|the)\s+)?["'“]?([^"'”?!.]{2,48})["'”]?(?:\s+space)?[?.!]*$/i,
  );
  if (addToSpaceMatch) {
    return {
      toolName: 'studio_manage_spaces',
      args: {
        action: 'add_item',
        artifact: addToSpaceMatch[1]?.trim() || 'latest',
        space: addToSpaceMatch[2]!.trim().replace(/\s+space$/i, ''),
      },
      confidenceQ10: 955,
      reason: 'Add artifact to Space Prism.',
    };
  }

  // 4. Open / Delete / Publish a Prism or Space by name
  const manageMatch = trimmed.match(
    /^(?:please\s+)?(open|delete|remove|publish)\s+(?:(?:my|the)\s+)?(?:prism|space|app|game|picture|track|document)?\s*["'“]?([^"'”?!.]{2,56})["'”]?[?.!]*$/i,
  );
  if (manageMatch && !/\bnotes?\b/i.test(trimmed)) {
    const verb = manageMatch[1]!.toLowerCase();
    const action = verb === 'remove' ? 'delete' : verb;
    return {
      toolName: 'studio_manage_spaces',
      args: { action, target: manageMatch[2]!.trim() },
      confidenceQ10: 945,
      reason: `${action} Prism "${manageMatch[2]!.trim()}".`,
    };
  }

  // 5. Refine an existing Prism ("Update <name> to ...", "Make that picture night with stars")
  const refineMatch = trimmed.match(
    /^(?:please\s+)?(?:update|edit|refine|change|modify|remix)\s+(?:(?:my|the)\s+)?(?:(?:last|latest|that|this)\s+(?:picture|image|artwork|audio|song|track|video|game|document|story|space|prism)|["'“]([^"'”]{2,56})["'”]|([a-z0-9][a-z0-9\s-]{1,40}))\s+(?:to|with|into|so\s+it)\s+(.+)$/i,
  );
  if (refineMatch) {
    const target = (refineMatch[1] ?? refineMatch[2] ?? 'latest').trim();
    const instruction = refineMatch[3]!.trim();
    return {
      toolName: 'studio_refine_prism',
      args: { target, instruction },
      confidenceQ10: 950,
      reason: `Refine Prism "${target}" via Verdant Studio.`,
    };
  }

  // 6. Direct natural-language multimedia creation via Verdant Studio
  const isCreationVerb =
    /^(?:please\s+)?(?:generate|create|make|draw|paint|render|design|compose|build|produce|synthesize|craft|write)\b/i.test(
      trimmed,
    );
  if (!isCreationVerb) return null;

  const explicitMedia =
    /\b(picture|image|artwork|painting|illustration|poster|wallpaper|portrait|landscape|nebula|seascape|pixel\s+art|mosaic|constellation|audio|music|song|soundtrack|soundscape|beat|lo-?fi|melody|synth|sound\s+effects?|video|animation|motion\s+graphic|storyboard|reel|trailer|game|minesweeper|breakout|pong|snake|2048|memory\s+match|lights?\s+out|connect\s*4|tic[\s-]?tac[\s-]?toe|gomoku|game\s+of\s+life|rule\s*30|novel|short\s+story|manuscript|chapter|research\s+program|study\s+plan|habit\s+tracker|habit\s+guardian|schedule\s+guard|workflow\s+checklist|calculator\s+app)\b/i.test(
      trimmed,
    );

  if (
    /^write\b/i.test(trimmed) &&
    !/\b(novel|book|short\s+story|manuscript|chapters?|screenplay|research\s+program)\b/i.test(trimmed)
  ) {
    return null;
  }

  if (explicitMedia) {
    const forSpace = trimmed.match(/\b(?:for|in|into)\s+(?:(?:my|the)\s+)?["'“]?([A-Z][a-zA-Z0-9\s-]{1,36})["'”]?(?:\s+space)?$/);
    const spaceName = forSpace?.[1]?.trim();
    return {
      toolName: 'studio_create_artifact',
      args: {
        prompt: trimmed,
        ...(spaceName ? { spaceName } : {}),
      },
      confidenceQ10: 960,
      reason: 'Synthesize multimedia artifact through Verdant Studio.',
    };
  }

  return null;
}
