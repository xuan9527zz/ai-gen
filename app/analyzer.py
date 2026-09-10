# -*- coding: utf-8 -*-
"""
Illustrious Image Analyzer / Orchestrator V2.5.2

V2.5.2 user-confirmed mapping provenance:
- Merge local Pixiv tag overrides above the public Japanese mapping.
- Record the raw-tag-to-prompt mapping source in every new Analysis Run.

V2.5.1 visual-style retention:
- Requires the merger to cover every explicitly described high-level category.
- Adds conservative deterministic recovery for rendering, material, lighting,
  depth-of-field and atmosphere phrases that are explicit in the VLM caption.
- Rejects rule matches in negated contexts such as "no bokeh".
- Records model-extracted and deterministically recovered features separately.

V2.4.0 Pixiv unknown-tag verification:
- WD14 remains the conservative base evidence.
- VLM contributes ONLY high-level visual features.
- Qwen Merger may REPORT conflicts/disagreements, but it has NO deletion authority.
- Python removes only deterministic censorship tags for an uncensored target.
- Unknown WD14 tags are preserved.
- Hierarchy/implication cleanup is disabled for multi-subject scenes.
- VLM visual features are category-gated and hard-filtered before entering the final prompt.
- Keeps all V2.2.2 behavior and Final Prompt logic unchanged.
- Re-checks subject-count disagreements deterministically in Python.
- If VLM explicitly says 1 woman + 2 men and WD14 says 1girl + 2boys,
  the disagreement is automatically removed even if Qwen called it strong.
- If VLM claims complete visibility but a strong unresolved subject-count
  disagreement remains, visibility is downgraded to uncertain.
- During unresolved subject-count disagreement, participant-evidence tags
  such as multiple penises / testicles / male-participant tags are protected
  from omission-only false-positive accusations.
- REVIEW NEEDED remains diagnostic only and never changes the final prompt.
- Source tags can now be supplied from Danbooru / Pixiv / raw text.
- Danbooru copied tag lists are cleaned by source_tag_adapter.py.
- Pixiv Japanese tags are mapped through jp_to_danbooru_tags.json.
- Unknown Pixiv Japanese tags are sent to a local semantic-mapping LLM.
- Candidate English tags are exact-match checked against RaidenInfinity's NAIDv3 Tag Search.
- NAID Tag Suggest-backed exact matches are auto-added; fallback/unverified candidates are preserved but not silently added.
- Final prompt base is now: Source Tags + WD14 Tags + accepted high-level VLM visual tags.
- Source Tags are preserved as independent provenance in each JSON run record.
- Partial/off-frame participant detection now requires HUMAN-context evidence.
- Source-backed tags are protected from omission-only false-positive reports.
- Final Prompt composition remains unchanged from V2.3.0.

Pipeline:
    Image
      ├─> Qwen3-VL NSFW Caption V4.5
      ├─> ComfyUI WD14
      └─> Source Tags
            ├─ Danbooru copied tags -> clean/normalize
            └─ Pixiv Japanese tags -> mapped Danbooru-style tags
               ↓
          Qwen3 8B Merger
        (analysis/reporting only)
               ↓
          Python Normalizer
               ↓
          Final Illustrious Prompt

Requirements:
- Ollama running at http://127.0.0.1:11434
- ComfyUI running at http://127.0.0.1:8188
- F:\\OpenWebUI\\wd14_api.json exists
- Python package: requests

Batch runner compatibility:
- Existing batch_analyze.py can continue importing:
      from illustrious_orchestrator import analyze_image
"""

import base64
import json
import os
import re
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

from .source_tags import (
    ADAPTER_VERSION as SOURCE_ADAPTER_VERSION,
    adapt_source_tags,
)

from .naid_verifier import (
    VERIFIER_VERSION as NAID_VERIFIER_VERSION,
    resolve_pixiv_unknown_tags,
)

from .image_inputs import prepare_wd14_image


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLLAMA_URL = os.getenv(
    "ILLUSTRIOUS_OLLAMA_URL",
    "http://127.0.0.1:11434",
)
COMFY_URL = os.getenv(
    "ILLUSTRIOUS_COMFY_URL",
    "http://127.0.0.1:8188",
)

VLM_MODEL = os.getenv(
    "ILLUSTRIOUS_VLM_MODEL",
    (
        "hf.co/mradermacher/"
        "Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF:Q5_K_M"
    ),
)

MERGER_MODEL = os.getenv(
    "ILLUSTRIOUS_MERGER_MODEL",
    "qwen3:8b",
)

WD14_WORKFLOW = str(
    PROJECT_ROOT
    / "workflows"
    / "wd14_api.json"
)

WD14_LOAD_IMAGE_NODE = "3"
WD14_OUTPUT_NODE = "4"

VLM_CONTEXT = 16384
MERGER_CONTEXT = 8192

VLM_NUM_PREDICT = 900
MERGER_NUM_PREDICT = 1100

RUN_PARALLEL = True

TARGET_ADULT = True
TARGET_UNCENSORED = True

SAVE_RUN_JSON = True
RUNS_DIR = PROJECT_ROOT / "data" / "runs"

JP_TAG_MAPPING = PROJECT_ROOT / "config" / "jp_to_danbooru_tags.json"

PIXIV_UNKNOWN_VERIFY = True
PIXIV_TRANSLATION_MODEL = os.getenv(
    "ILLUSTRIOUS_PIXIV_MODEL",
    "qwen3:8b",
)

MAX_VISUAL_TAG_WORDS = 5


# ============================================================
# VLM PROMPT
# ============================================================

VLM_PROMPT = """
Analyze this image carefully for image-generation prompting.

Describe only visually supported information.

Focus on:
- visible subject count and spatial arrangement
- subject appearance
- hairstyle, clothing and accessories
- facial expression
- pose, gesture and action
- composition and framing
- camera angle and perspective
- background and environment
- lighting and shadows
- colors and color relationships
- materials and surface appearance
- depth of field, blur, bokeh, reflections and other visual effects
- illustration and rendering style
- small visible details

Do not guess:
- original prompt
- LoRA
- trigger words
- checkpoint
- seed
- CFG
- sampler
- artist name

Do not infer youth or an underage age category from appearance.

Give a detailed visual caption.
""".strip()


# ============================================================
# STRUCTURED MERGER SYSTEM PROMPT
# ============================================================

MERGER_SYSTEM_PROMPT = r"""
You are the evidence-comparison layer in an Illustrious image-analysis pipeline.

IMPORTANT:
You are NOT allowed to delete, replace, or rewrite WD14 tags or Source Tags.
Python will build the final prompt conservatively.

Your job is only to:
1. extract high-level visual features from the VLM caption,
2. report major disagreements between VLM, WD14, and Source Tags,
3. report genuine same-attribute conflicts,
4. flag suspicious WD14 tags for later human review,
5. report uncertain tags.

INPUT SOURCES

1. VLM Caption
An independent natural-language visual description.

2. WD14 Auto Tags
Independent Danbooru-style tags.

3. Source Tags
Optional source metadata.
- Danbooru: cleaned source-page tags.
- Pixiv: only Japanese tags that have been normalized/mapped to English Danbooru-style tags.
Source Tags are independent evidence and must not be silently deleted.

4. Target Instructions
The intended output target.

SOURCE ROLES

Source Tags are high-value source metadata when present.
They may contain:
- character / copyright tags
- subject count
- appearance
- clothing
- accessories
- pose/action
- scene attributes

Do not assume Source Tags are wrong merely because the VLM caption omitted them.
Do not use Source Tags as high-level visual_features; Python adds them separately.

WD14 is the PRIMARY source for:
- subject count tags
- hair
- eyes
- clothing
- accessories
- facial expression
- pose
- gestures
- anatomy/body visibility
- adult anatomy and adult actions
- fine-grained Danbooru-style attributes

VLM is the PRIMARY source for:
- composition
- framing
- camera angle
- perspective
- foreground/background relationship
- scene/background
- depth of field
- blur / bokeh
- lighting
- shadows
- color palette
- atmosphere
- material/surface appearance
- rendering / illustration style

HIGH-LEVEL VISUAL FEATURES ONLY

visual_features must use one of these categories exactly:
- composition
- camera
- background
- lighting
- color
- atmosphere
- depth_of_field
- rendering
- material

Each visual feature tag must:
- be English
- be concise
- ideally be 1 to 5 words
- not be a sentence
- not merely restate WD14 body/clothing/hair/pose attributes

CATEGORY COVERAGE IS REQUIRED:
- Inspect the complete VLM caption before finishing visual_features.
- For every allowed category explicitly described by the VLM, include at least
  one discriminative feature from that category.
- In particular, do not omit rendering or material features when the caption
  describes painterly / realistic / cel-shaded rendering, surface texture,
  fabric texture, glossy surfaces, or similar appearance cues.
- Prefer 1 to 3 useful features per explicitly described category.
- Preserve meaningful qualifiers. For example, use "painterly rendering" or
  "realistic illustration", not the vague tag "detailed style".
- Do not stop after a fixed total number of features.

GOOD examples:
{"category":"composition","tag":"centered composition"}
{"category":"camera","tag":"close-up"}
{"category":"background","tag":"classroom background"}
{"category":"background","tag":"dark wooden panels"}
{"category":"lighting","tag":"warm lighting"}
{"category":"lighting","tag":"dramatic shadows"}
{"category":"color","tag":"vibrant color palette"}
{"category":"depth_of_field","tag":"shallow depth of field"}
{"category":"depth_of_field","tag":"bokeh"}
{"category":"material","tag":"glossy skin"}
{"category":"rendering","tag":"soft shading"}

BAD visual features:
- red hair
- purple eyes
- large breasts
- black bra
- school uniform
- spread legs
- sitting
- handjob
- open mouth
- nipple piercing
- gold bikini
- uncensored
- adult nsfw

Those are not high-level visual features.

PARTIAL / OFF-FRAME PARTICIPANTS

Subject counting must account for cropped and partially visible participants.

A participant may be present even when only part of the body is visible, such as:
- a hand or arm entering the frame
- a torso at the edge
- a partially cropped body
- a face or head cut off by the frame
- genitals or another body part belonging to an off-frame participant
- interaction evidence clearly caused by another person outside the central crop

Do NOT equate "one central character" with "only one participant".

If the VLM caption indicates partial or off-frame people, use:
participant_visibility.status = "partial_or_offframe"

If the visible participant structure cannot be confidently counted, use:
participant_visibility.status = "uncertain"

Use "complete" only when the scene appears to show the relevant participants clearly enough
for subject counting to be reasonably reliable.

MAJOR DISAGREEMENTS

Explicitly compare:
- apparent subject count
- fully visible vs partially visible / off-frame participants
- apparent gender/role count if visible
- major interaction structure
- major scene structure

If VLM and WD14 strongly disagree, report it in major_disagreements.
Do not resolve it by deleting tags.

IMPORTANT:
When participants are cropped, partially visible, or off-frame,
WD14 subject-count / interaction tags may be more reliable than the VLM's
natural-language count.

Example:
VLM describes one central adult woman but also mentions a male arm or other
partially visible participant, while WD14 contains "2boys", "multiple boys",
"group sex". This should be REVIEWED, not automatically treated as a WD14 error.

CONFLICTS

A genuine conflict exists ONLY when two claims describe the SAME visual attribute
and cannot reasonably both be true.

Examples:
- red hair vs blue hair
- sitting vs standing
- looking at viewer vs looking away
- indoor vs outdoor

These are NOT conflicts:
- eye color + pupil shape
- gaze direction + hand gesture
- sweat + wet skin
- bikini + navel piercing
- school uniform + breasts out
- arms raised + open clothes
- a broad VLM clothing description + a more specific WD14 clothing tag

Do not invent conflicts just to fill the field.

SUSPECTED FALSE POSITIVES

suspected_false_positives may list WD14 tags that have strong contradictory visual evidence.

This field is for REVIEW ONLY.
Do not assume they will be deleted.

A tag being absent from the VLM caption is NOT sufficient evidence.

SPECIAL RULE FOR SOURCE TAGS:
If a tag is present in Source Tags, do NOT call it a suspected false positive
merely because the VLM did not mention it or only implied it.
Source metadata is independent evidence.

SPECIAL RULE FOR PARTICIPANT / INTERACTION TAGS:
Do NOT mark the following kinds of WD14 tags as false positives merely because the
VLM did not mention all participants:
- 2boys / 3boys / multiple boys
- 2girls / 3girls / multiple girls
- group sex
- threesome / foursome / orgy
- other subject-count or interaction-structure tags

This is especially important when participant_visibility is partial_or_offframe
or uncertain.

Only flag such a structural tag when there is strong direct contradictory evidence,
not simple omission from the VLM caption.

ADULT CONTENT

The target workflow may involve clearly adult NSFW imagery.

Do NOT suppress or downgrade visually supported adult tags merely because
they are sexual, nude, explicit, or adult-oriented.

All sexual subjects must be adults.
Never introduce youth-related, school-age, teen, child, underage,
or ambiguous-age sexual descriptors.

UNCENSORED TARGET

If the target is uncensored, censorship-related WD14 tags can be reported,
but Python will handle their deterministic removal.

Do NOT output:
- uncensored
- adult nsfw
- nsfw
as visual features.

OUTPUT

Return VALID JSON ONLY.
No Markdown.
No code fences.
No prose before or after JSON.

Use exactly this schema:

{
  "visual_features": [
    {
      "category": "lighting",
      "tag": "warm lighting"
    }
  ],
  "participant_visibility": {
    "status": "complete",
    "evidence": []
  },
  "major_disagreements": [
    {
      "attribute": "subject_count",
      "vlm": "one central adult woman",
      "wd14": ["2boys", "multiple boys"],
      "assessment": "strong disagreement"
    }
  ],
  "conflicts": [
    {
      "attribute": "hair_color",
      "vlm": ["blue hair"],
      "wd14": ["red hair"],
      "assessment": "genuine conflict"
    }
  ],
  "suspected_false_positives": [
    {
      "tag": "tag",
      "reason": "brief evidence-based reason"
    }
  ],
  "uncertain_tags": [
    "tag"
  ],
  "notes": []
}

participant_visibility.status must be exactly one of:
- complete
- partial_or_offframe
- uncertain

participant_visibility.evidence:
Short evidence phrases only.
Use [] if there is nothing useful to report.

Return [] for list fields with nothing to report.
""".strip()


# ============================================================
# DETERMINISTIC NORMALIZER RULES
# ============================================================

TAG_ALIASES = {
    # Keep conservative.
    # Add only aliases that are certainly equivalent.
}

# specific_tag -> more general tags
#
# IMPORTANT:
# These rules are applied ONLY when Python believes the image is
# a single-subject scene. In multi-subject scenes, hierarchy
# cleanup is disabled because general and specific tags may
# legitimately refer to different subjects.
TAG_IMPLICATIONS = {
    "micro bikini": {
        "bikini",
        "swimsuit",
    },
    "leopard print": {
        "animal print",
    },
    "navel piercing": {
        "piercing",
        "navel",
    },
    "heart-shaped pupils": {
        "symbol-shaped pupils",
    },
}

CENSORSHIP_TAGS = {
    "censored",
    "mosaic censoring",
    "bar censor",
    "light censor",
    "heart censor",
    "steam censor",
    "convenient censoring",
    "black censor",
    "white censor",
}

ALLOWED_VISUAL_CATEGORIES = {
    "composition",
    "camera",
    "background",
    "lighting",
    "color",
    "atmosphere",
    "depth_of_field",
    "rendering",
    "material",
}

META_VISUAL_TAGS = {
    "uncensored",
    "adult",
    "adult nsfw",
    "nsfw",
    "explicit",
    "explicit content",
    "rating explicit",
    "rating questionable",
}


STRUCTURAL_PARTICIPANT_TAGS = {
    "multiple boys",
    "multiple girls",
    "multiple others",
    "group sex",
    "threesome",
    "foursome",
    "orgy",
    "mmf threesome",
    "mff threesome",
    "ffm threesome",
    "mfm threesome",
}


PARTICIPANT_EVIDENCE_TAGS = {
    "penis",
    "multiple penises",
    "testicles",
    "erection",
    "dark-skinned male",
    "male",
    "male focus",
    "female",
    "female focus",
    "man",
    "men",
    "woman",
    "women",
    "boy",
    "boys",
    "girl",
    "girls",
}


def is_participant_evidence_tag(tag):
    """
    Tags that can indicate the presence of an additional participant,
    even when that participant is cropped or mostly off-frame.
    """
    tag = canonicalize_tag(tag)

    if is_structural_participant_tag(
        tag
    ):
        return True

    if tag in PARTICIPANT_EVIDENCE_TAGS:
        return True

    # Conservative pattern support for common participant terms.
    if re.fullmatch(
        r"(dark-skinned )?(male|female)",
        tag,
    ):
        return True

    if re.fullmatch(
        r"(multiple )?penises?",
        tag,
    ):
        return True

    return False

PARTIAL_PARTICIPANT_DIRECT_HINTS = (
    "cropped participant",
    "off-frame participant",
    "off frame participant",
    "only the hand",
    "only a hand",
    "only the arm",
    "only an arm",
    "part of a male",
    "part of a female",
    "part of a man",
    "part of a woman",
    "male arm entering the frame",
    "male hand entering the frame",
    "female arm entering the frame",
    "female hand entering the frame",
)

PARTICIPANT_NOUN_PATTERN = (
    r"(?:male|female|man|woman|men|women|boy|girl|boys|girls|"
    r"person|people|figure|figures|participant|participants|"
    r"character|characters|body|bodies|hand|hands|arm|arms|"
    r"torso|torsos|head|heads|face|faces)"
)

PARTIAL_STATE_PATTERN = (
    r"(?:partially visible|partly visible|partially shown|partly shown|"
    r"cropped|cut off|off-frame|off frame|entering the frame|"
    r"at the edge of the frame)"
)


def is_structural_participant_tag(tag):
    tag = canonicalize_tag(tag)

    if tag in STRUCTURAL_PARTICIPANT_TAGS:
        return True

    if re.fullmatch(
        r"([2-9]|[1-9][0-9]+)(boys?|girls?|others?)",
        tag,
    ):
        return True

    return False


def reason_is_omission_based(reason):
    """
    Heuristic for explanations that are basically:
    "the VLM did not mention / did not see it".

    Those are too weak to reject structural participant tags.
    """
    reason = str(reason).strip().lower()

    markers = (
        "not mentioned",
        "not explicitly mentioned",
        "not described",
        "not directly stated",
        "implied but not directly stated",
        "no mention",
        "no visible",
        "not visible",
        "no male figures",
        "no female figures",
        "no males",
        "no females",
        "no indication",
        "no evidence",
        "not implied",
        "not present in vlm",
        "vlm caption",
        "vlm description",
    )

    return any(
        marker in reason
        for marker in markers
    )


def find_partial_participant_evidence(caption):
    """
    Return HUMAN-context partial/off-frame evidence.

    Ignore generic object phrases such as:
    - "window partially visible"
    - "genital area partially obscured"
    - "background partially blurred"

    Keep participant phrases such as:
    - "two male figures are partially visible"
    - "partially visible male figure"
    - "cropped female body"
    - "only a hand is visible"
    """
    caption = str(caption or "").lower()

    evidence = []

    for hint in PARTIAL_PARTICIPANT_DIRECT_HINTS:
        if hint in caption:
            evidence.append(
                hint
            )

    pattern_after = re.compile(
        rf"\b{PARTICIPANT_NOUN_PATTERN}\b"
        rf"(?:\s+\w+){{0,5}}\s+"
        rf"{PARTIAL_STATE_PATTERN}"
    )

    pattern_before = re.compile(
        rf"{PARTIAL_STATE_PATTERN}"
        rf"(?:\s+\w+){{0,5}}\s+"
        rf"\b{PARTICIPANT_NOUN_PATTERN}\b"
    )

    for regex in (
        pattern_after,
        pattern_before,
    ):
        for match in regex.finditer(
            caption
        ):
            phrase = " ".join(
                match.group(0).split()
            )

            if phrase not in evidence:
                evidence.append(
                    phrase
                )

    return evidence


def caption_has_partial_participant_hints(caption):
    return bool(
        find_partial_participant_evidence(
            caption
        )
    )

# Terms strongly indicating that a VLM "visual feature" is actually
# a subject/body/clothing/pose tag and therefore belongs to WD14,
# not the high-level visual layer.
LOW_LEVEL_VISUAL_TERMS = {
    "hair",
    "eyes",
    "eye",
    "iris",
    "pupils",
    "pupil",
    "breast",
    "breasts",
    "nipples",
    "nipple",
    "penis",
    "testicles",
    "pussy",
    "vagina",
    "anus",
    "pubic",
    "bikini",
    "bra",
    "panties",
    "thong",
    "shirt",
    "skirt",
    "dress",
    "uniform",
    "jacket",
    "choker",
    "earrings",
    "earring",
    "piercing",
    "tattoo",
    "tail",
    "animal ears",
    "gloves",
    "shoe",
    "shoes",
    "heels",
    "mouth",
    "tongue",
    "smile",
    "expression",
    "blush",
    "sitting",
    "standing",
    "kneeling",
    "squatting",
    "lying",
    "spread legs",
    "arms up",
    "hands",
    "hand",
    "fingering",
    "handjob",
    "nude",
    "topless",
    "bottomless",
    "cum",
    "saliva",
    "pregnant",
    "lactation",
    "one-eyed",
    "eyepatch",
    "muscular arms",
}

# Generic caption-like filler that is not useful enough to justify
# adding another prompt token.
GENERIC_VISUAL_FILLER = {
    "highly detailed realistic style",
    "highly detailed style",
    "detailed illustration style",
    "clean illustration style",
    "realistic style",
    "high detail textures",
    "highly detailed",
    "very detailed",
}


# Conservative caption-to-feature recovery. These rules do not attempt to
# reinterpret the scene; they only preserve high-level phrases explicitly
# present in the VLM caption when the merger model omits them.
CAPTION_VISUAL_RULES = (
    (
        "rendering",
        "semi-realistic illustration",
        (r"\bsemi[- ]realistic\b",),
    ),
    (
        "rendering",
        "photorealistic rendering",
        (r"\bphoto[- ]?realistic\b", r"\bphotorealism\b"),
    ),
    (
        "rendering",
        "realistic illustration",
        (
            r"\b(?:illustration|art|rendering) style\b[^.]{0,80}\brealistic\b",
            r"\brealistic (?:illustration|rendering|art style)\b",
        ),
    ),
    (
        "rendering",
        "highly detailed illustration",
        (r"\bhighly detailed\b", r"\brich in detail\b"),
    ),
    (
        "rendering",
        "painterly rendering",
        (r"\bpainterly\b",),
    ),
    (
        "rendering",
        "digital painting",
        (r"\bdigital painting\b", r"\bdigitally painted\b"),
    ),
    (
        "rendering",
        "cel shading",
        (r"\bcel[- ]shad(?:ed|ing)\b",),
    ),
    (
        "rendering",
        "anime illustration",
        (r"\banime (?:style|illustration|aesthetic)\b",),
    ),
    (
        "rendering",
        "watercolor illustration",
        (r"\bwatercolou?r\b",),
    ),
    (
        "rendering",
        "oil painting",
        (r"\boil painting\b",),
    ),
    (
        "rendering",
        "3d rendering",
        (r"\b3d render(?:ing|ed)?\b",),
    ),
    (
        "rendering",
        "soft shading",
        (r"\bsoft shading\b",),
    ),
    (
        "material",
        "detailed fabric texture",
        (
            r"\bfabric texture(?:s)?\b",
            r"\btexture(?:s)? of (?:the )?fabric\b",
        ),
    ),
    (
        "material",
        "intricate lace texture",
        (r"\b(?:intricate|delicate) lace (?:pattern|texture)(?:s)?\b",),
    ),
    (
        "material",
        "woven texture",
        (
            r"\bwoven texture\b",
            r"\btexture of (?:the )?woven\b",
        ),
    ),
    (
        "material",
        "glossy skin",
        (r"\bglossy skin\b", r"\bskin (?:appears|looks) glossy\b"),
    ),
    (
        "material",
        "subsurface scattering",
        (r"\bsubsurface scattering\b",),
    ),
    (
        "lighting",
        "soft diffused lighting",
        (
            r"\bsoft(?: and)? diffused light(?:ing)?\b",
            r"\blighting is soft and diffused\b",
        ),
    ),
    (
        "lighting",
        "warm golden lighting",
        (r"\bwarm(?:,| and)? golden (?:light|lighting|glow)\b",),
    ),
    (
        "lighting",
        "volumetric lighting",
        (r"\bvolumetric light(?:ing)?\b",),
    ),
    (
        "lighting",
        "soft bloom",
        (r"\bsoft bloom\b",),
    ),
    (
        "lighting",
        "subtle shadows",
        (r"\bsubtle shadows\b",),
    ),
    (
        "lighting",
        "dramatic shadows",
        (r"\bdramatic shadows\b",),
    ),
    (
        "lighting",
        "rim lighting",
        (r"\brim light(?:ing)?\b",),
    ),
    (
        "lighting",
        "backlighting",
        (r"\bback[- ]?light(?:ing|lit)?\b",),
    ),
    (
        "depth_of_field",
        "shallow depth of field",
        (r"\bshallow depth of field\b",),
    ),
    (
        "depth_of_field",
        "slightly blurred background",
        (
            r"\bslightly blurred background\b",
            r"\bbackground (?:is |remains )?slightly blurred\b",
        ),
    ),
    (
        "depth_of_field",
        "blurred background",
        (
            r"(?<!slightly )(?<!softly )\bblurred background\b",
            r"\bbackground (?:is |remains )?blurred\b",
        ),
    ),
    (
        "depth_of_field",
        "bokeh",
        (r"\bbokeh\b",),
    ),
    (
        "atmosphere",
        "ethereal atmosphere",
        (r"\bethereal atmosphere\b",),
    ),
    (
        "atmosphere",
        "dreamy atmosphere",
        (r"\bdreamy atmosphere\b",),
    ),
)


# ============================================================
# BASIC HELPERS
# ============================================================

def canonicalize_tag(tag):
    tag = str(tag).strip().lower()
    tag = tag.replace("_", " ")
    tag = " ".join(tag.split())
    return TAG_ALIASES.get(tag, tag)


def exact_deduplicate(tags):
    result = []
    seen = set()

    for raw_tag in tags:
        tag = canonicalize_tag(raw_tag)

        if not tag:
            continue

        if tag not in seen:
            seen.add(tag)
            result.append(tag)

    return result


def parse_wd14_tags(text):
    raw = str(text).replace("\n", " ")
    tags = []

    for item in raw.split(","):
        tag = canonicalize_tag(item)
        if tag:
            tags.append(tag)

    return exact_deduplicate(tags)


def detect_multi_subject_scene(tags):
    """
    Conservative detector.

    If subject structure appears multi-person/multi-character,
    implication cleanup is disabled.
    """
    tags = set(exact_deduplicate(tags))

    direct_multi_tags = {
        "multiple girls",
        "multiple boys",
        "multiple others",
        "group sex",
        "threesome",
        "foursome",
        "orgy",
    }

    if tags & direct_multi_tags:
        return True

    # 2girls, 3girls, 2boys, 3boys, etc.
    count_pattern = re.compile(
        r"^([2-9]|[1-9][0-9]+)(girls?|boys?|others?)$"
    )

    for tag in tags:
        if count_pattern.match(tag):
            return True

    return False


def apply_implications(tags, multi_subject=False):
    """
    Apply hierarchy cleanup only for single-subject scenes.
    """
    tags = exact_deduplicate(tags)

    if multi_subject:
        return tags

    active = set(tags)

    for specific, generals in TAG_IMPLICATIONS.items():
        if specific in active:
            for general in generals:
                active.discard(general)

    return [
        tag
        for tag in tags
        if tag in active
    ]


def is_short_prompt_phrase(tag):
    tag = canonicalize_tag(tag)

    if not tag:
        return False

    if "," in tag:
        return False

    if any(ch in tag for ch in ".;:!?()[]{}"):
        return False

    words = tag.split()

    if len(words) < 1 or len(words) > MAX_VISUAL_TAG_WORDS:
        return False

    sentence_markers = {
        "while",
        "which",
        "that",
        "who",
        "whose",
        "where",
        "wearing",
        "holding",
        "pulling",
        "showing",
        "revealing",
        "positioned",
        "placed",
    }

    if any(word in sentence_markers for word in words):
        return False

    if not re.fullmatch(r"[a-z0-9\s/'\-]+", tag):
        return False

    return True


def _normalize_match_text(text):
    """
    Normalize text for whole-word / whole-phrase matching.

    Examples:
    - "vibrant color palette" must NOT match "bra"
    - "detailed anime style" must NOT match "tail"
    - "black bra" SHOULD match "bra"
    - "animal ears" SHOULD match "animal ears"
    """
    text = canonicalize_tag(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _contains_whole_phrase(text, phrase):
    """
    Whole token/phrase match instead of raw substring matching.
    """
    text_norm = _normalize_match_text(text)
    phrase_norm = _normalize_match_text(phrase)

    if not text_norm or not phrase_norm:
        return False

    text_tokens = text_norm.split()
    phrase_tokens = phrase_norm.split()

    if len(phrase_tokens) == 1:
        return phrase_tokens[0] in text_tokens

    size = len(phrase_tokens)

    for i in range(
        len(text_tokens) - size + 1
    ):
        if (
            text_tokens[i:i + size]
            == phrase_tokens
        ):
            return True

    return False


def contains_low_level_term(tag):
    """
    V2.2.1:
    Use whole-word / whole-phrase matching.

    This fixes false matches such as:
    - "bra" inside "vibrant"
    - "tail" inside "detailed"
    """
    tag = canonicalize_tag(tag)

    for term in LOW_LEVEL_VISUAL_TERMS:
        if _contains_whole_phrase(
            tag,
            term,
        ):
            return True

    return False


def extract_caption_visual_features(caption):
    """Recover explicit high-level visual cues from a VLM caption."""

    text = " ".join(
        str(caption or "").lower().split()
    )
    features = []

    if not text:
        return features

    for category, tag, patterns in CAPTION_VISUAL_RULES:
        match = None

        for pattern in patterns:
            for candidate in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                if not caption_match_is_negated(
                    text,
                    candidate,
                ):
                    match = candidate
                    break
            if match is not None:
                break

        if not match:
            continue

        features.append({
            "category": category,
            "tag": tag,
            "origin": "vlm_caption_rule",
            "evidence": " ".join(
                match.group(0).split()
            ),
        })

    return features


def caption_match_is_negated(text, match):
    """Detect a local explicit negation around one caption rule match."""

    left = text[
        max(0, match.start() - 100):match.start()
    ]
    right = text[
        match.end():min(len(text), match.end() + 50)
    ]
    left_clause = re.split(
        r"[.,;!?]|\bbut\b|\bhowever\b",
        left,
    )[-1]

    if re.search(
        r"\b(?:no|not|without|lacks?|lacking|absence of)\b"
        r"[^.;]{0,80}$",
        left_clause,
    ):
        return True

    if re.match(
        r"\s+(?:effects? )?(?:is |are )?"
        r"(?:absent|missing|not visible)\b",
        right,
    ):
        return True

    return False


def visual_feature_dedupe_key(item):
    category = str(
        item.get("category", "")
    ).strip().lower()
    tag = canonicalize_tag(
        item.get("tag", "")
    )

    if category == "rendering":
        tag = re.sub(
            r"\s+style$",
            "",
            tag,
        )

    return category, tag


def combine_visual_features(*feature_groups):
    """Combine feature groups while preserving first-source provenance."""

    combined = []
    seen = set()

    for group in feature_groups:
        if not isinstance(group, list):
            continue

        for item in group:
            if not isinstance(item, dict):
                combined.append(item)
                continue

            key = visual_feature_dedupe_key(
                item
            )

            if key in seen:
                continue

            seen.add(key)
            combined.append(item)

    return combined


def sanitize_visual_features(features):
    """
    Strong V2.2 gate.

    A VLM feature enters the final prompt only when:
    - it is an object with an allowed high-level category,
    - it is a short prompt-like phrase,
    - it is not target/meta information,
    - it is not generic filler,
    - it does not look like subject/body/clothing/pose detail.

    Returns:
      accepted_tags
      accepted_features
      rejected_features
    """
    accepted_tags = []
    accepted_features = []
    rejected_features = []

    if not isinstance(features, list):
        return [], [], []

    for item in features:
        if not isinstance(item, dict):
            rejected_features.append({
                "item": item,
                "reason": "feature is not an object",
            })
            continue

        category = str(
            item.get("category", "")
        ).strip().lower()

        tag = canonicalize_tag(
            item.get("tag", "")
        )

        if category not in ALLOWED_VISUAL_CATEGORIES:
            rejected_features.append({
                "category": category,
                "tag": tag,
                "reason": "category not allowed",
            })
            continue

        if not tag:
            rejected_features.append({
                "category": category,
                "tag": tag,
                "reason": "empty tag",
            })
            continue

        if tag in META_VISUAL_TAGS:
            rejected_features.append({
                "category": category,
                "tag": tag,
                "reason": "target/meta tag",
            })
            continue

        if tag in GENERIC_VISUAL_FILLER:
            rejected_features.append({
                "category": category,
                "tag": tag,
                "reason": "generic filler",
            })
            continue

        if not is_short_prompt_phrase(tag):
            rejected_features.append({
                "category": category,
                "tag": tag,
                "reason": "not a short prompt phrase",
            })
            continue

        if contains_low_level_term(tag):
            rejected_features.append({
                "category": category,
                "tag": tag,
                "reason": "low-level subject/body/clothing/pose detail",
            })
            continue

        accepted_tags.append(tag)
        accepted_feature = {
            "category": category,
            "tag": tag,
        }

        origin = str(
            item.get("origin", "")
        ).strip()
        evidence = str(
            item.get("evidence", "")
        ).strip()

        if origin:
            accepted_feature["origin"] = origin
        if evidence:
            accepted_feature["evidence"] = evidence

        accepted_features.append(
            accepted_feature
        )

    accepted_tags = exact_deduplicate(
        accepted_tags
    )

    # Deduplicate accepted feature objects by final tag.
    dedup_features = []
    seen = set()

    for feature in accepted_features:
        tag = feature["tag"]

        if tag not in seen:
            seen.add(tag)
            dedup_features.append(feature)

    return (
        accepted_tags,
        dedup_features,
        rejected_features,
    )


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


# ============================================================
# VLM
# ============================================================

def run_vlm(image_path):
    print("[VLM] 开始分析图片...")
    start = time.time()

    image_b64 = image_to_base64(
        image_path
    )

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": VLM_PROMPT,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": VLM_CONTEXT,
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": VLM_NUM_PREDICT,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=1200,
    )
    response.raise_for_status()

    data = response.json()

    caption = (
        data.get("message", {})
        .get("content", "")
        .strip()
    )

    if not caption:
        raise RuntimeError(
            "VLM 返回了空内容。"
        )

    elapsed = time.time() - start

    print(
        f"[VLM] 完成，用时 "
        f"{elapsed:.1f} 秒"
    )

    return caption


# ============================================================
# WD14 / COMFYUI
# ============================================================

def upload_to_comfy(image_path):
    image_path = Path(image_path)

    with open(image_path, "rb") as f:
        files = {
            "image": (
                image_path.name,
                f,
                "application/octet-stream",
            )
        }

        data = {
            "type": "input",
            "overwrite": "true",
        }

        response = requests.post(
            f"{COMFY_URL}/upload/image",
            files=files,
            data=data,
            timeout=60,
        )

    response.raise_for_status()
    result = response.json()

    if "name" not in result:
        raise RuntimeError(
            f"ComfyUI 上传返回异常：{result}"
        )

    return result["name"]


def load_wd14_workflow():
    path = Path(WD14_WORKFLOW)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到 WD14 workflow：{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        workflow = json.load(f)

    if WD14_LOAD_IMAGE_NODE not in workflow:
        raise KeyError(
            f"WD14 workflow 中没有节点 "
            f"{WD14_LOAD_IMAGE_NODE}"
        )

    return workflow


def queue_comfy_workflow(workflow):
    payload = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }

    response = requests.post(
        f"{COMFY_URL}/prompt",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()

    if "prompt_id" not in result:
        raise RuntimeError(
            f"ComfyUI 提交 workflow "
            f"返回异常：{result}"
        )

    return result["prompt_id"]


def get_comfy_history(prompt_id):
    response = requests.get(
        f"{COMFY_URL}/history/{prompt_id}",
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def extract_wd14_tags(
    history,
    prompt_id,
):
    if prompt_id not in history:
        return None

    outputs = (
        history[prompt_id]
        .get("outputs", {})
    )

    possible_fields = [
        "text",
        "string",
        "output",
        "value",
        "values",
    ]

    node = outputs.get(
        WD14_OUTPUT_NODE,
        {},
    )

    for field in possible_fields:
        if field in node:
            value = node[field]

            if isinstance(value, list):
                return "\n".join(
                    str(x)
                    for x in value
                )

            return str(value)

    # Fallback: inspect all output nodes.
    for node_data in outputs.values():
        for field in possible_fields:
            if field in node_data:
                value = node_data[field]

                if isinstance(value, list):
                    return "\n".join(
                        str(x)
                        for x in value
                    )

                return str(value)

    return None


def run_wd14_with_metadata(image_path):
    print("[WD14] 开始分析图片...")
    start = time.time()

    with prepare_wd14_image(
        image_path
    ) as (wd14_path, input_metadata):
        if input_metadata[
            "strategy"
        ] == "first_frame_png":
            print(
                "[WD14] 检测到动画图片："
                f"{input_metadata['source_format']} / "
                f"{input_metadata['source_frame_count']} frames；"
                "仅使用第 1 帧。"
            )

        uploaded_name = upload_to_comfy(
            wd14_path
        )

    workflow = load_wd14_workflow()

    workflow[
        WD14_LOAD_IMAGE_NODE
    ]["inputs"]["image"] = uploaded_name

    prompt_id = queue_comfy_workflow(
        workflow
    )

    while time.time() - start < 180:
        history = get_comfy_history(
            prompt_id
        )

        tags = extract_wd14_tags(
            history,
            prompt_id,
        )

        if tags:
            elapsed = time.time() - start

            print(
                f"[WD14] 完成，用时 "
                f"{elapsed:.1f} 秒"
            )

            return (
                tags.strip(),
                input_metadata,
            )

        time.sleep(1)

    raise TimeoutError(
        "WD14 处理超过 180 秒。"
    )


def run_wd14(image_path):
    """Backward-compatible WD14 entry point returning tags only."""

    tags, _metadata = run_wd14_with_metadata(
        image_path
    )
    return tags


# ============================================================
# STRUCTURED MERGER
# ============================================================

def extract_json_from_text(text):
    text = str(text).strip()

    if text.startswith("```"):
        text = text.replace(
            "```json",
            "",
            1,
        )
        text = text.replace(
            "```",
            "",
        )
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "Merger 没有返回有效 JSON。"
            "\n原始输出：\n"
            + text
        )

    return json.loads(
        text[start:end + 1]
    )


def validate_merger_data(data):
    if not isinstance(data, dict):
        raise ValueError(
            "Merger JSON 顶层必须是 object。"
        )

    participant_visibility = data.get(
        "participant_visibility",
        {},
    )

    if not isinstance(
        participant_visibility,
        dict,
    ):
        participant_visibility = {}

    visibility_status = str(
        participant_visibility.get(
            "status",
            "uncertain",
        )
    ).strip().lower()

    if visibility_status not in {
        "complete",
        "partial_or_offframe",
        "uncertain",
    }:
        visibility_status = "uncertain"

    visibility_evidence = participant_visibility.get(
        "evidence",
        [],
    )

    if not isinstance(
        visibility_evidence,
        list,
    ):
        visibility_evidence = []

    result = {
        "visual_features": data.get(
            "visual_features",
            [],
        ),
        "participant_visibility": {
            "status": visibility_status,
            "evidence": visibility_evidence,
        },
        "major_disagreements": data.get(
            "major_disagreements",
            [],
        ),
        "conflicts": data.get(
            "conflicts",
            [],
        ),
        "suspected_false_positives": data.get(
            "suspected_false_positives",
            [],
        ),
        "uncertain_tags": data.get(
            "uncertain_tags",
            [],
        ),
        "notes": data.get(
            "notes",
            [],
        ),
    }

    list_fields = (
        "visual_features",
        "major_disagreements",
        "conflicts",
        "suspected_false_positives",
        "uncertain_tags",
        "notes",
    )

    for key in list_fields:
        if not isinstance(
            result[key],
            list,
        ):
            result[key] = []

    return result


def extract_wd14_evidence(item):
    """
    Robustly recover WD evidence even if the LLM emits malformed keys
    such as wd13, wd1, wd, wd_14, etc.

    The canonical downstream key is always wd14.
    """
    if not isinstance(
        item,
        dict,
    ):
        return []

    direct = item.get(
        "wd14",
        None,
    )

    if direct not in (
        None,
        "",
        [],
    ):
        if isinstance(
            direct,
            list,
        ):
            return direct
        return [direct]

    # Search malformed WD-like keys.
    for key, value in item.items():
        key_norm = re.sub(
            r"[^a-z0-9]",
            "",
            str(key).lower(),
        )

        if (
            key_norm == "wd"
            or key_norm.startswith("wd1")
            or key_norm.startswith("wd")
        ):
            if key_norm in {
                "wd14autotags",
                "wd14tags",
            }:
                pass

            if value in (
                None,
                "",
                [],
            ):
                continue

            if isinstance(
                value,
                list,
            ):
                return value

            return [value]

    return []


def normalize_wd14_keys(items):
    """
    Add/repair canonical wd14 field in diagnostic objects.
    Raw model keys remain available elsewhere for audit.
    """
    if not isinstance(
        items,
        list,
    ):
        return []

    normalized = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        clean = dict(item)

        evidence = extract_wd14_evidence(
            item
        )

        if evidence:
            clean["wd14"] = evidence

        normalized.append(
            clean
        )

    return normalized


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _number_token_to_int(token):
    token = str(token).strip().lower()

    if token.isdigit():
        return int(token)

    return NUMBER_WORDS.get(
        token
    )


def infer_vlm_gender_counts(text):
    """
    Extract explicit gendered participant counts from natural-language
    VLM text.

    Missing gender counts remain UNKNOWN, not zero.

    Examples:
    - "two nude female characters" -> female: 2
    - "one central adult woman, two male figures partially visible"
      -> female: 1, male: 2

    V2.2.3 implementation note:
    token scanning is used instead of a broad regex so that
    "one central woman, two male figures" cannot be misread as
    "one male".
    """
    text = str(text).lower()

    clean = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    tokens = clean.split()

    female_tokens = {
        "female",
        "woman",
        "women",
        "girl",
        "girls",
    }

    male_tokens = {
        "male",
        "man",
        "men",
        "boy",
        "boys",
    }

    # Descriptor words that can appear between the number and noun.
    # We search backward up to four tokens and stop if we hit another
    # explicit gender noun.
    counts_found = {
        "female": [],
        "male": [],
    }

    for i, token in enumerate(
        tokens
    ):
        if token in female_tokens:
            gender = "female"
        elif token in male_tokens:
            gender = "male"
        else:
            continue

        start = max(
            0,
            i - 5,
        )

        value = None

        for j in range(
            i - 1,
            start - 1,
            -1,
        ):
            prev = tokens[j]

            # Do not cross another gender noun.
            if (
                prev in female_tokens
                or prev in male_tokens
            ):
                break

            parsed = _number_token_to_int(
                prev
            )

            if parsed is not None:
                value = parsed
                break

        if value is not None:
            counts_found[
                gender
            ].append(
                value
            )

    counts = {}

    for gender, values in counts_found.items():
        if values:
            counts[gender] = max(
                values
            )

    return counts


def infer_wd14_gender_counts(tags):
    """
    Extract exact gendered counts from WD14 evidence.

    Exact tags such as 1girl / 2boys are used.
    "multiple boys" is structural evidence but not an exact numeric count.
    """
    tags = [
        canonicalize_tag(tag)
        for tag in tags
    ]

    counts = {}

    for tag in tags:
        match = re.fullmatch(
            r"(\d+)(girls?|boys?)",
            tag,
        )

        if not match:
            continue

        number = int(
            match.group(1)
        )

        noun = match.group(2)

        if noun.startswith(
            "girl"
        ):
            gender = "female"
        else:
            gender = "male"

        # Keep the strongest exact count if duplicated.
        if (
            gender not in counts
            or number > counts[gender]
        ):
            counts[gender] = number

    return counts


def subject_counts_explicitly_agree(
    vlm_text,
    wd14_evidence,
):
    """
    True only when every exact WD14 gender count is also explicitly
    present in the VLM text and matches.

    This avoids falsely declaring agreement for:
        VLM: "one central woman"
        WD14: 1girl + 2boys
    because the VLM did NOT explicitly count the men.

    But it correctly recognizes:
        VLM: "one woman, two men partially visible"
        WD14: 1girl + 2boys
    """
    vlm_counts = infer_vlm_gender_counts(
        vlm_text
    )

    wd_counts = infer_wd14_gender_counts(
        wd14_evidence
    )

    if not wd_counts:
        return False

    # Every exact WD dimension must be explicitly represented by VLM.
    for gender, wd_count in wd_counts.items():
        if gender not in vlm_counts:
            return False

        if vlm_counts[gender] != wd_count:
            return False

    return True


def reconcile_subject_count_disagreements(
    items,
):
    """
    Remove false subject-count disagreements when Python can prove that
    the explicit VLM count and exact WD14 count actually agree.

    All unresolved cases remain for REVIEW.
    """
    if not isinstance(
        items,
        list,
    ):
        return []

    cleaned = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        attribute = str(
            item.get(
                "attribute",
                "",
            )
        ).strip().lower()

        if attribute not in {
            "subject_count",
            "subject count",
            "participant_count",
            "participant count",
        }:
            cleaned.append(
                item
            )
            continue

        vlm_text = item.get(
            "vlm",
            "",
        )

        wd14_evidence = extract_wd14_evidence(
            item
        )

        if subject_counts_explicitly_agree(
            vlm_text,
            wd14_evidence,
        ):
            # Qwen called it a disagreement, but deterministic
            # count parsing proves the counts actually align.
            continue

        cleaned.append(
            item
        )

    return cleaned


def has_strong_subject_count_disagreement(
    items,
):
    if not isinstance(
        items,
        list,
    ):
        return False

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        attribute = str(
            item.get(
                "attribute",
                "",
            )
        ).strip().lower()

        assessment = _assessment_text(
            item
        )

        if (
            attribute
            in {
                "subject_count",
                "subject count",
                "participant_count",
                "participant count",
            }
            and (
                "strong disagreement"
                in assessment
                or "major disagreement"
                in assessment
                or "significant disagreement"
                in assessment
            )
        ):
            return True

    return False


def downgrade_complete_visibility_if_needed(
    participant_visibility,
    major_disagreements,
):
    """
    If VLM claims visibility is complete but an unresolved strong
    subject-count disagreement remains, do not trust "complete".

    Downgrade to "uncertain" rather than guessing which model is right.
    """
    visibility = dict(
        participant_visibility
        if isinstance(
            participant_visibility,
            dict,
        )
        else {}
    )

    status = str(
        visibility.get(
            "status",
            "uncertain",
        )
    ).strip().lower()

    evidence = visibility.get(
        "evidence",
        [],
    )

    if not isinstance(
        evidence,
        list,
    ):
        evidence = []

    if (
        status == "complete"
        and has_strong_subject_count_disagreement(
            major_disagreements
        )
    ):
        status = "uncertain"

        note = (
            "VLM claimed complete visibility, but a strong unresolved "
            "WD14/VLM subject-count disagreement remains"
        )

        if note not in evidence:
            evidence.append(
                note
            )

    visibility[
        "status"
    ] = status

    visibility[
        "evidence"
    ] = evidence

    return visibility



def clean_suspected_false_positives(
    items,
    participant_visibility,
    vlm_caption,
    major_disagreements=None,
    source_tags=None,
):
    """
    V2.2.2:
    Structural participant/interaction tags are NOT treated as false
    positives merely because the VLM omitted partially visible people.

    This is diagnostic-only. Final prompt logic is unchanged.
    """
    if not isinstance(
        items,
        list,
    ):
        return []

    visibility_status = str(
        participant_visibility.get(
            "status",
            "uncertain",
        )
    ).strip().lower()

    partial_hint = (
        visibility_status
        in {
            "partial_or_offframe",
            "uncertain",
        }
        or caption_has_partial_participant_hints(
            vlm_caption
        )
    )

    strong_subject_count_disagreement = (
        has_strong_subject_count_disagreement(
            major_disagreements
            if major_disagreements is not None
            else []
        )
    )

    source_tag_set = set(
        exact_deduplicate(
            source_tags
            or []
        )
    )

    cleaned = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        tag = canonicalize_tag(
            item.get(
                "tag",
                "",
            )
        )

        reason = str(
            item.get(
                "reason",
                "",
            )
        )

        if (
            tag in source_tag_set
            and reason_is_omission_based(
                reason
            )
        ):
            # Source-page metadata is independent evidence.
            # VLM omission alone is not enough to reject it.
            continue

        if (
            is_structural_participant_tag(
                tag
            )
            and reason_is_omission_based(
                reason
            )
        ):
            # Structural participant tags are never rejected merely
            # because the VLM failed to mention all participants.
            continue

        if (
            strong_subject_count_disagreement
            and is_participant_evidence_tag(
                tag
            )
            and reason_is_omission_based(
                reason
            )
        ):
            # During unresolved participant-count disagreement,
            # protect evidence such as penis / multiple penises /
            # testicles / male-participant tags from omission-only
            # false-positive accusations.
            continue

        cleaned.append(
            item
        )

    return cleaned



def _assessment_text(item):
    if not isinstance(item, dict):
        return ""

    return str(
        item.get(
            "assessment",
            "",
        )
    ).strip().lower()


def is_non_conflict_diagnostic(item):
    """
    True when the model itself explicitly says the item is NOT a conflict.

    We only clean the display/diagnostic layer.
    This does NOT alter WD14 tags or the final prompt.
    """
    assessment = _assessment_text(
        item
    )

    if not assessment:
        return False

    non_conflict_markers = (
        "no disagreement",
        "no conflict",
        "no genuine conflict",
        "not a conflict",
        "consistent",
        "same attribute, consistent",
        "partial agreement",
        "agreement",
    )

    return any(
        marker in assessment
        for marker in non_conflict_markers
    )


def clean_conflicts(items):
    """
    Remove entries that explicitly describe agreement / consistency.

    Genuine or uncertain conflicts stay visible.
    """
    if not isinstance(
        items,
        list,
    ):
        return []

    cleaned = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if is_non_conflict_diagnostic(
            item
        ):
            continue

        cleaned.append(
            item
        )

    return cleaned


def clean_major_disagreements(items):
    """
    Keep only actual disagreement entries.

    Entries saying "no disagreement" are removed.
    """
    if not isinstance(
        items,
        list,
    ):
        return []

    cleaned = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        assessment = _assessment_text(
            item
        )

        if (
            "no disagreement"
            in assessment
        ):
            continue

        cleaned.append(
            item
        )

    return cleaned


def build_review_status(
    major_disagreements,
):
    """
    REVIEW NEEDED is triggered only by strong structural disagreement.

    This is deliberately conservative:
    it does not delete any WD14 tag.
    """
    reasons = []

    for item in major_disagreements:
        if not isinstance(
            item,
            dict,
        ):
            continue

        assessment = _assessment_text(
            item
        )

        if (
            "strong disagreement"
            in assessment
            or "major disagreement"
            in assessment
            or "significant disagreement"
            in assessment
        ):
            reasons.append({
                "attribute": item.get(
                    "attribute",
                    "",
                ),
                "assessment": item.get(
                    "assessment",
                    "",
                ),
                "vlm": item.get(
                    "vlm",
                    "",
                ),
                "wd14": extract_wd14_evidence(
                    item
                ),
            })

    return (
        bool(reasons),
        reasons,
    )



def run_merger(
    vlm_caption,
    wd14_tags,
    source_type="none",
    source_tags=None,
):
    print(
        "[MERGER] 开始证据比较..."
    )

    start = time.time()

    target_text = "adult NSFW"

    if TARGET_UNCENSORED:
        target_text += ", uncensored"

    user_prompt = f"""
Target:
{target_text}

VLM Caption:
{vlm_caption}

WD14 Auto Tags:
{wd14_tags}

Source Type:
{source_type}

Source Tags:
{", ".join(source_tags or []) if source_tags else "None"}

User Modifications:
None

Compare the evidence sources.
Do not delete or rewrite WD14 tags or Source Tags.
Return structured JSON only.
""".strip()

    payload = {
        "model": MERGER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": MERGER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "num_ctx": MERGER_CONTEXT,
            "temperature": 0.12,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": MERGER_NUM_PREDICT,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()

    raw_text = (
        response.json()
        .get("message", {})
        .get("content", "")
        .strip()
    )

    merger_data = extract_json_from_text(
        raw_text
    )

    merger_data = validate_merger_data(
        merger_data
    )

    # --------------------------------------------------------
    # V2.2.1 diagnostic cleanup
    # --------------------------------------------------------
    #
    # Keep the model's raw diagnostics for audit, but remove
    # entries that explicitly say there is no disagreement /
    # no genuine conflict from the active display fields.
    # --------------------------------------------------------

    merger_data[
        "major_disagreements_raw"
    ] = list(
        merger_data[
            "major_disagreements"
        ]
    )

    merger_data[
        "conflicts_raw"
    ] = list(
        merger_data[
            "conflicts"
        ]
    )

    merger_data[
        "suspected_false_positives_raw"
    ] = list(
        merger_data[
            "suspected_false_positives"
        ]
    )

    # Canonicalize malformed wd13 / wd1 / wd keys.
    merger_data[
        "major_disagreements"
    ] = normalize_wd14_keys(
        merger_data[
            "major_disagreements"
        ]
    )

    merger_data[
        "conflicts"
    ] = normalize_wd14_keys(
        merger_data[
            "conflicts"
        ]
    )

    # If the model failed to mark partial visibility but the caption
    # explicitly contains strong partial-participant language,
    # downgrade subject-count confidence conservatively.
    participant_visibility = merger_data[
        "participant_visibility"
    ]

    partial_participant_evidence = (
        find_partial_participant_evidence(
            vlm_caption
        )
    )

    if (
        participant_visibility.get(
            "status"
        )
        == "complete"
        and partial_participant_evidence
    ):
        participant_visibility[
            "status"
        ] = "partial_or_offframe"

        evidence_list = (
            participant_visibility.setdefault(
                "evidence",
                [],
            )
        )

        for evidence in (
            partial_participant_evidence
        ):
            if evidence not in evidence_list:
                evidence_list.append(
                    evidence
                )

    merger_data[
        "participant_visibility"
    ] = participant_visibility

    merger_data[
        "major_disagreements"
    ] = clean_major_disagreements(
        merger_data[
            "major_disagreements"
        ]
    )

    # --------------------------------------------------------
    # V2.2.3 deterministic subject-count reconciliation
    # --------------------------------------------------------
    #
    # Example:
    # VLM: 1 woman + 2 partially visible men
    # WD14: 1girl + 2boys
    #
    # If Qwen still calls this "strong disagreement", Python
    # removes that false disagreement.
    # --------------------------------------------------------

    merger_data[
        "major_disagreements"
    ] = reconcile_subject_count_disagreements(
        merger_data[
            "major_disagreements"
        ]
    )

    # If "complete" visibility conflicts with unresolved strong
    # participant-count evidence, downgrade to uncertain.
    participant_visibility = (
        downgrade_complete_visibility_if_needed(
            participant_visibility,
            merger_data[
                "major_disagreements"
            ],
        )
    )

    merger_data[
        "participant_visibility"
    ] = participant_visibility

    merger_data[
        "conflicts"
    ] = clean_conflicts(
        merger_data[
            "conflicts"
        ]
    )

    merger_data[
        "suspected_false_positives"
    ] = clean_suspected_false_positives(
        merger_data[
            "suspected_false_positives"
        ],
        participant_visibility=(
            participant_visibility
        ),
        vlm_caption=vlm_caption,
        major_disagreements=(
            merger_data[
                "major_disagreements"
            ]
        ),
        source_tags=(
            source_tags
            or []
        ),
    )

    (
        review_needed,
        review_reasons,
    ) = build_review_status(
        merger_data[
            "major_disagreements"
        ]
    )

    merger_data[
        "review_needed"
    ] = review_needed

    merger_data[
        "review_reasons"
    ] = review_reasons

    # Keep model and deterministic VLM features separately for audit.
    model_visual_features = list(
        merger_data["visual_features"]
    )
    deterministic_visual_features = (
        extract_caption_visual_features(
            vlm_caption
        )
    )
    raw_visual_features = combine_visual_features(
        model_visual_features,
        deterministic_visual_features,
    )

    merger_data[
        "visual_features_model_raw"
    ] = model_visual_features

    merger_data[
        "visual_features_deterministic"
    ] = deterministic_visual_features

    (
        accepted_tags,
        accepted_features,
        rejected_features,
    ) = sanitize_visual_features(
        raw_visual_features
    )

    merger_data[
        "visual_features_raw"
    ] = raw_visual_features

    merger_data[
        "visual_features"
    ] = accepted_features

    merger_data[
        "visual_tags_accepted"
    ] = accepted_tags

    merger_data[
        "visual_features_rejected"
    ] = rejected_features

    elapsed = time.time() - start

    print(
        f"[MERGER] 完成，用时 "
        f"{elapsed:.1f} 秒"
    )

    return merger_data


# ============================================================
# PYTHON NORMALIZER
# ============================================================

def normalize_prompt_data(
    wd14_text,
    merger_data,
    source_tags=None,
    target_uncensored=True,
):
    """
    V2.3 conservative policy:

    - Base = normalized Source Tags + all WD14 tags.
    - Merger has NO deletion authority.
    - Only deterministic censorship tags are removed for uncensored target.
    - Conflicts are report-only.
    - Suspected false positives are report-only.
    - VLM contributes only accepted high-level visual tags.
    - Multi-subject scene -> hierarchy/implication pruning is disabled.
    """

    wd14_tags = parse_wd14_tags(
        wd14_text
    )

    normalized_source_tags = (
        exact_deduplicate(
            source_tags
            or []
        )
    )

    combined_evidence_tags = (
        normalized_source_tags
        + wd14_tags
    )

    multi_subject = (
        detect_multi_subject_scene(
            combined_evidence_tags
        )
    )

    removed = set()

    # --------------------------------------------------------
    # 1. Deterministic censorship removal only
    # --------------------------------------------------------

    if target_uncensored:
        for tag in combined_evidence_tags:
            if tag in CENSORSHIP_TAGS:
                removed.add(tag)

    source_tags_used = [
        tag
        for tag in normalized_source_tags
        if tag not in removed
    ]

    wd14_tags_used = [
        tag
        for tag in wd14_tags
        if tag not in removed
    ]

    # Source tags come first so source-page metadata keeps priority
    # in ordering, while exact duplicates from WD14 are removed later.
    base_tags = (
        source_tags_used
        + wd14_tags_used
    )

    # --------------------------------------------------------
    # 2. Optional hierarchy cleanup
    # --------------------------------------------------------

    base_tags = apply_implications(
        base_tags,
        multi_subject=multi_subject,
    )

    # --------------------------------------------------------
    # 3. Accepted high-level VLM visual tags
    # --------------------------------------------------------

    visual_tags = exact_deduplicate(
        merger_data.get(
            "visual_tags_accepted",
            [],
        )
    )

    final_tags = exact_deduplicate(
        base_tags + visual_tags
    )

    # --------------------------------------------------------
    # 4. Uncertain tags are review-only and should not be
    #    duplicated if they are already in final prompt.
    # --------------------------------------------------------

    final_set = set(
        final_tags
    )

    uncertain = exact_deduplicate(
        merger_data.get(
            "uncertain_tags",
            [],
        )
    )

    uncertain = [
        tag
        for tag in uncertain
        if tag not in final_set
    ]

    return {
        "final_tags": final_tags,
        "source_tags_used": exact_deduplicate(
            source_tags_used
        ),
        "visual_tags_added": visual_tags,
        "removed_tags": sorted(
            removed
        ),
        "multi_subject_detected": (
            multi_subject
        ),
        "implication_cleanup_applied": (
            not multi_subject
        ),
        "participant_visibility": merger_data.get(
            "participant_visibility",
            {
                "status": "uncertain",
                "evidence": [],
            },
        ),
        "review_needed": merger_data.get(
            "review_needed",
            False,
        ),
        "review_reasons": merger_data.get(
            "review_reasons",
            [],
        ),
        "major_disagreements": merger_data.get(
            "major_disagreements",
            [],
        ),
        "conflicts": merger_data.get(
            "conflicts",
            [],
        ),
        "suspected_false_positives": merger_data.get(
            "suspected_false_positives",
            [],
        ),
        "uncertain_tags": uncertain,
        "notes": merger_data.get(
            "notes",
            [],
        ),
        "rejected_visual_features": merger_data.get(
            "visual_features_rejected",
            [],
        ),
    }


# ============================================================
# RUN RECORD
# ============================================================

def save_run_record(result):
    if not SAVE_RUN_JSON:
        return None

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    image_stem = (
        Path(result["image"])
        .stem
        .replace(" ", "_")
    )

    output_path = (
        RUNS_DIR
        / f"{timestamp}_{image_stem}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ============================================================
# PRINT HELPERS
# ============================================================

def print_json_or_none(data):
    if data:
        print(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("None")


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_image(
    image_path,
    source_type="none",
    source_tags_text="",
    source_mapping_path=JP_TAG_MAPPING,
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"找不到图片：{image_path}"
        )

    source_info = adapt_source_tags(
        source_type=source_type,
        source_text=source_tags_text,
        mapping_path=source_mapping_path,
    )

    source_tags = list(
        source_info.get(
            "normalized_tags",
            [],
        )
    )

    pixiv_verification = {
        "verifier_version": (
            NAID_VERIFIER_VERSION
        ),
        "translation_model": (
            PIXIV_TRANSLATION_MODEL
        ),
        "verification_source": (
            "https://raideninfinity.pythonanywhere.com/naidv3_tag_search"
        ),
        "records": [],
        "auto_added_tags": [],
        "suggested_tags": [],
        "unverified_source_tags": [],
        "status": "not_applicable",
    }

    if (
        source_info.get(
            "source_type"
        )
        == "pixiv"
        and source_info.get(
            "unknown_tags"
        )
        and PIXIV_UNKNOWN_VERIFY
    ):
        try:
            pixiv_verification = (
                resolve_pixiv_unknown_tags(
                    source_info.get(
                        "unknown_tags",
                        [],
                    ),
                    translation_model=(
                        PIXIV_TRANSLATION_MODEL
                    ),
                )
            )

            pixiv_verification[
                "status"
            ] = "completed"

            for tag in pixiv_verification.get(
                "auto_added_tags",
                [],
            ):
                if tag not in source_tags:
                    source_tags.append(
                        tag
                    )

        except Exception as exc:
            pixiv_verification = {
                "verifier_version": (
                    NAID_VERIFIER_VERSION
                ),
                "translation_model": (
                    PIXIV_TRANSLATION_MODEL
                ),
                "verification_source": (
                    "https://raideninfinity.pythonanywhere.com/naidv3_tag_search"
                ),
                "records": [],
                "auto_added_tags": [],
                "suggested_tags": [],
                "unverified_source_tags": (
                    source_info.get(
                        "unknown_tags",
                        [],
                    )
                ),
                "status": "error",
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    print()
    print("=" * 60)
    print(
        "Illustrious Image Analyzer V2.5.2"
    )
    print("=" * 60)
    print(f"图片：{image_path}")
    print()

    total_start = time.time()

    # --------------------------------------------------------
    # 1. VLM + WD14 independent analysis
    # --------------------------------------------------------

    if RUN_PARALLEL:
        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            vlm_future = (
                executor.submit(
                    run_vlm,
                    str(image_path),
                )
            )

            wd14_future = (
                executor.submit(
                    run_wd14_with_metadata,
                    str(image_path),
                )
            )

            vlm_caption = (
                vlm_future.result()
            )

            (
                wd14_tags,
                wd14_input,
            ) = (
                wd14_future.result()
            )

    else:
        vlm_caption = run_vlm(
            str(image_path)
        )

        (
            wd14_tags,
            wd14_input,
        ) = run_wd14_with_metadata(
            str(image_path),
        )

    # --------------------------------------------------------
    # 2. Source tags
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SOURCE TAGS")
    print("=" * 60)

    print(
        f"source_type: "
        f"{source_info.get('source_type', 'none')}"
    )

    if source_tags:
        print(
            ", ".join(
                source_tags
            )
        )
    else:
        print("None")

    unknown_source_tags = source_info.get(
        "unknown_tags",
        [],
    )

    if unknown_source_tags:
        print()
        print(
            "PIXIV UNKNOWN TAGS:"
        )
        print(
            ", ".join(
                unknown_source_tags
            )
        )

        if (
            source_info.get(
                "source_type"
            )
            == "pixiv"
        ):
            auto_added = (
                pixiv_verification.get(
                    "auto_added_tags",
                    [],
                )
            )

            suggested = (
                pixiv_verification.get(
                    "suggested_tags",
                    [],
                )
            )

            print()
            print(
                "PIXIV VERIFIED AUTO-ADDED:"
            )
            print(
                ", ".join(
                    auto_added
                )
                if auto_added
                else "None"
            )

            print()
            print(
                "PIXIV SUGGESTED / NOT AUTO-ADDED:"
            )
            print(
                ", ".join(
                    suggested
                )
                if suggested
                else "None"
            )

    # --------------------------------------------------------
    # 3. Raw vision sources
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VLM CAPTION")
    print("=" * 60)
    print(vlm_caption)

    print()
    print("=" * 60)
    print("WD14 TAGS")
    print("=" * 60)
    print(wd14_tags)

    # --------------------------------------------------------
    # 3. Merger = report-only evidence comparison
    # --------------------------------------------------------

    print()

    merger_data = run_merger(
        vlm_caption,
        wd14_tags,
        source_type=(
            source_info.get(
                "source_type",
                "none",
            )
        ),
        source_tags=source_tags,
    )

    # --------------------------------------------------------
    # 4. Python normalizer
    # --------------------------------------------------------

    normalized = normalize_prompt_data(
        wd14_text=wd14_tags,
        merger_data=merger_data,
        source_tags=source_tags,
        target_uncensored=(
            TARGET_UNCENSORED
        ),
    )

    final_tags = normalized[
        "final_tags"
    ]

    final_prompt = ", ".join(
        final_tags
    )

    # --------------------------------------------------------
    # 5. Structured merger result
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "STRUCTURED MERGER RESULT"
    )
    print("=" * 60)

    print(
        json.dumps(
            merger_data,
            ensure_ascii=False,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 6. Source tags used in final prompt
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SOURCE TAGS ADDED")
    print("=" * 60)

    if normalized[
        "source_tags_used"
    ]:
        print(
            ", ".join(
                normalized[
                    "source_tags_used"
                ]
            )
        )
    else:
        print("None")

    # --------------------------------------------------------
    # 7. Accepted VLM visual tags
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VLM VISUAL TAGS ADDED")
    print("=" * 60)

    if normalized[
        "visual_tags_added"
    ]:
        print(
            ", ".join(
                normalized[
                    "visual_tags_added"
                ]
            )
        )
    else:
        print("None")

    # --------------------------------------------------------
    # 7. Rejected VLM visual features
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "VLM VISUAL FEATURES REJECTED"
    )
    print("=" * 60)

    print_json_or_none(
        normalized[
            "rejected_visual_features"
        ]
    )

    # --------------------------------------------------------
    # 8. Final prompt
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "FINAL ILLUSTRIOUS PROMPT"
    )
    print("=" * 60)
    print(final_prompt)

    # --------------------------------------------------------
    # 9. Deterministic removals
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("REMOVED TAGS")
    print("=" * 60)

    if normalized[
        "removed_tags"
    ]:
        print(
            ", ".join(
                normalized[
                    "removed_tags"
                ]
            )
        )
    else:
        print("None")

    # --------------------------------------------------------
    # 10. Multi-subject / implication status
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("NORMALIZER STATUS")
    print("=" * 60)

    print(
        "multi_subject_detected: "
        f"{normalized['multi_subject_detected']}"
    )

    print(
        "implication_cleanup_applied: "
        f"{normalized['implication_cleanup_applied']}"
    )

    # --------------------------------------------------------
    # 11. Participant visibility
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PARTICIPANT VISIBILITY")
    print("=" * 60)

    print_json_or_none(
        normalized[
            "participant_visibility"
        ]
    )

    # --------------------------------------------------------
    # 12. Review status
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("REVIEW NEEDED")
    print("=" * 60)

    if normalized[
        "review_needed"
    ]:
        print("True")
        print()
        print("Strong structural disagreement detected:")
        print_json_or_none(
            normalized[
                "review_reasons"
            ]
        )
    else:
        print("False")

    # --------------------------------------------------------
    # 13. Major disagreements
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MAJOR DISAGREEMENTS")
    print("=" * 60)

    print_json_or_none(
        normalized[
            "major_disagreements"
        ]
    )

    # --------------------------------------------------------
    # 14. Genuine conflicts
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CONFLICTS")
    print("=" * 60)

    print_json_or_none(
        normalized[
            "conflicts"
        ]
    )

    # --------------------------------------------------------
    # 15. Suspected false positives
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "SUSPECTED FALSE POSITIVES"
    )
    print("=" * 60)

    print_json_or_none(
        normalized[
            "suspected_false_positives"
        ]
    )

    # --------------------------------------------------------
    # 16. Uncertain
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "CANDIDATE / UNCERTAIN TAGS"
    )
    print("=" * 60)

    if normalized[
        "uncertain_tags"
    ]:
        print(
            ", ".join(
                normalized[
                    "uncertain_tags"
                ]
            )
        )
    else:
        print("None")

    # --------------------------------------------------------
    # 17. Timing
    # --------------------------------------------------------

    total_elapsed = (
        time.time()
        - total_start
    )

    print()
    print(
        f"全部完成，总用时："
        f"{total_elapsed:.1f} 秒"
    )

    # --------------------------------------------------------
    # 18. Record
    # --------------------------------------------------------

    result = {
        "image": str(image_path),
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "version": "2.5.2",
        "models": {
            "vlm": VLM_MODEL,
            "merger": MERGER_MODEL,
        },
        "target": {
            "adult": TARGET_ADULT,
            "uncensored": (
                TARGET_UNCENSORED
            ),
        },
        "source": {
            "adapter_version": (
                SOURCE_ADAPTER_VERSION
            ),
            "source_type": source_info.get(
                "source_type",
                "none",
            ),
            "source_tags_raw": source_info.get(
                "source_tags_raw",
                "",
            ),
            "source_tags_raw_list": source_info.get(
                "source_tags_raw_list",
                [],
            ),
            "normalized_tags": source_tags,
            "known_mapped_tags": source_info.get(
                "normalized_tags",
                [],
            ),
            "mapping_records": source_info.get(
                "mapping_records",
                [],
            ),
            "unknown_tags": source_info.get(
                "unknown_tags",
                [],
            ),
            "pixiv_verification": (
                pixiv_verification
            ),
        },
        "vlm_caption": vlm_caption,
        "wd14_tags": wd14_tags,
        "wd14_input": wd14_input,
        "merger_data": merger_data,
        "normalizer": {
            "multi_subject_detected": (
                normalized[
                    "multi_subject_detected"
                ]
            ),
            "implication_cleanup_applied": (
                normalized[
                    "implication_cleanup_applied"
                ]
            ),
        },
        "source_tags_added": normalized[
            "source_tags_used"
        ],
        "visual_tags_added": normalized[
            "visual_tags_added"
        ],
        "rejected_visual_features": normalized[
            "rejected_visual_features"
        ],
        "final_tags": final_tags,
        "final_prompt": final_prompt,
        "removed_tags": normalized[
            "removed_tags"
        ],
        "participant_visibility": normalized[
            "participant_visibility"
        ],
        "review_needed": normalized[
            "review_needed"
        ],
        "review_reasons": normalized[
            "review_reasons"
        ],
        "major_disagreements": normalized[
            "major_disagreements"
        ],
        "conflicts": normalized[
            "conflicts"
        ],
        "suspected_false_positives": normalized[
            "suspected_false_positives"
        ],
        "uncertain_tags": normalized[
            "uncertain_tags"
        ],
        "notes": normalized[
            "notes"
        ],
        "elapsed_seconds": round(
            total_elapsed,
            2,
        ),
    }

    saved_path = save_run_record(
        result
    )

    if saved_path:
        print(
            f"运行记录已保存："
            f"{saved_path}"
        )

    return result


# ============================================================
# SOURCE TAG CLI HELPERS
# ============================================================

def read_multiline_source_tags():
    print()
    print(
        "请粘贴来源 tags。"
    )
    print(
        "Danbooru 可以直接粘贴带 [?]、链接、词条数量的整段文本。"
    )
    print(
        "Pixiv 可以粘贴日文 tags。"
    )
    print(
        "粘贴完后，单独输入 END 并回车。"
    )
    print()

    lines = []

    while True:
        line = input()

        if (
            line.strip().upper()
            == "END"
        ):
            break

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    print()
    print(
        "请输入图片完整路径："
    )

    image_path = (
        input("> ")
        .strip()
        .strip('"')
    )

    print()
    print(
        "来源类型："
    )
    print(
        "  none     = 没有来源 tags"
    )
    print(
        "  danbooru = Danbooru tags"
    )
    print(
        "  pixiv    = Pixiv 日文 tags"
    )
    print(
        "  raw      = 已经整理好的英文 tags"
    )

    source_type = (
        input(
            "source_type [none]: "
        )
        .strip()
        .lower()
        or "none"
    )

    if source_type == "none":
        source_tags_text = ""
    else:
        source_tags_text = (
            read_multiline_source_tags()
        )

    try:
        analyze_image(
            image_path,
            source_type=source_type,
            source_tags_text=(
                source_tags_text
            ),
        )

    except KeyboardInterrupt:
        print(
            "\n用户已取消。"
        )

    except Exception as e:
        print()
        print("=" * 60)
        print("发生错误")
        print("=" * 60)
        print(
            f"{type(e).__name__}: {e}"
        )
