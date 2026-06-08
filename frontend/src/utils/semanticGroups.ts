/**
 * Semantic subcategory grouping for General/Unknown tags.
 * Ported from image-tag-helper extension (info_panel.js SEMANTIC_GROUPS).
 */

interface SemanticGroupDef {
  name: string;
  pattern: RegExp;
}

const DEFS: SemanticGroupDef[] = [
  {
    name: "count",
    pattern: /\b(solo|duo|trio|\d+girls?|\d+boys?|\d+others?|multiple_girls|multiple_boys|group|everyone|_focus)\b/,
  },
  {
    name: "hair",
    pattern: /hair|twintail|braid|ponytail|ahoge|drill_hair|sidelocks|ringlets|bob_cut|hime_cut|pixie_cut|mohawk|blonde|brunette|redhead/,
  },
  {
    name: "face",
    pattern: /eye|mouth|lip|teeth|smile|blush|tongue|expression|head_tilt|bang|forehead|chin|cheek|nose|ear|eyebrow|makeup|lipstick|eyeshadow|mascara|eyeliner|rouge|tears|crying/,
  },
  {
    name: "body",
    pattern: /breast|boob|nipple|ass|butt|thigh|waist|belly|navel|stomach|skin|collarbone|shoulder|armpit|back|hip|arm\b|leg\b|feet|foot|barefoot|sole|toe|knee|finger|hand|wrist|elbow|ankle|calf|pubic|groin|neck|tail|wing|tentacle|fur|scale|claw|paw|feather/,
  },
  {
    name: "nudity",
    pattern: /\b(nude|naked|topless|bottomless|completely_nude|penis|pussy|vagina|vulva|anus|genitalia|sex\b|cum\b|orgasm|erect|uncensored|censored|mosaic)\b/,
  },
  {
    name: "underwear",
    pattern: /\b(bra|panties|thong|g-string|lingerie|underwear|panty|brassiere|sports_bra)\b/,
  },
  {
    name: "swimwear",
    pattern: /swimsuit|swimwear|racerback|highleg_swimsuit|leotard|competition_swimsuit|one-piece_swimsuit|bikini/,
  },
  {
    name: "top",
    pattern: /shirt|blouse|jacket|coat|sleeve|collar|vest|sweater|hoodie|uniform|dress|outfit|sideless|cutout|backless|bare_back|crop_top|midriff|kimono|yukata|cheongsam|maid|apron|cape|cloak|robe|bodysuit|jumpsuit/,
  },
  {
    name: "bottom",
    pattern: /\b(skirt|pants|shorts|leggings|jeans|trousers|hot_pants)\b/,
  },
  {
    name: "legwear",
    pattern: /boot|shoe|footwear|thighhigh|stocking|sock\b|pantyhose|kneehigh|zettai_ryouiki|slipper|sandal|heel|loafer|sneaker/,
  },
  {
    name: "handwear",
    pattern: /\b(gloves?|mittens?|gauntlets?)\b/,
  },
  {
    name: "headwear",
    pattern: /\b(hat|cap\b|helmet|crown|tiara|headband|hood|veil|bonnet|beret|headpiece|halo)\b/,
  },
  {
    name: "eyewear",
    pattern: /\b(glasses|sunglasses|eyewear|goggles|monocle)\b/,
  },
  {
    name: "acc",
    pattern: /ribbon|bowtie|bow_tie|necklace|earring|bracelet|ring\b|jewelry|hair_ornament|wand|belt|bag|umbrella|choker|pendant|brooch|anklet|nail|piercing|strap|lanyard|\bbow\b/,
  },
  {
    name: "weapon",
    pattern: /sword|blade|knife|dagger|katana|spear|lance|axe|hammer|gun|pistol|rifle|cannon|crossbow|shield|armor|weapon|explosive|grenade|magic_circle/,
  },
  {
    name: "object",
    pattern: /holding_|phone|book|flower|bouquet|fan|mirror|cup|glass|bottle|food|fruit|cake|candy|ice_cream|drink|coffee|tea|tray|camera|microphone|instrument|guitar|piano|laptop|controller|card|letter|envelope|scroll|flag|torch|lantern/,
  },
  {
    name: "animal",
    pattern: /\b(cat\b|dog\b|bird\b|rabbit|fox|wolf|horse|dragon|monster|animal|creature|pet\b|beast|feline|canine|animal_ears)\b/,
  },
  {
    name: "pose",
    pattern: /looking_|standing|sitting|lying|running|walking|cowboy_shot|upper_body|full_body|crouch|kneel|from_|spread|_pose|hiding|arms_up|arms_behind|legs_together|knees_up|arched|straddle|stretch|lean|bent_over|raised|outstretched|crossed_arms|hands_on|head_rest|tiptoe|jumping|floating|flying/,
  },
  {
    name: "bg",
    pattern: /background|outdoor|indoor|window|sky|grass|forest|water|ocean|sea|river|street|room|floor|wall|ceiling|city|town|urban|building|castle|school|beach|mountain|desert|space|planet|cloud|day|night|sunlight|moonlight|rain|snow|storm|fog|scenery|against_/,
  },
  {
    name: "lighting",
    pattern: /light|shadow|glow|gleam|sparkle|shiny|reflecti|backlight|rim_light|lens_flare|bloom|dark\b|bright\b|gradient|monochrome|greyscale|sepia|colorful|saturated/,
  },
  {
    name: "text",
    pattern: /speech_bubble|dialogue|text\b|watermark|signature|logo|symbol|caption|subtitle/,
  },
  {
    name: "style",
    pattern: /realistic|photorealistic|parody|cosplay|magical_girl|fantasy|blurry|blur\b|bokeh|depth_of_field|motion_blur|chromatic|sketch|lineart|flat_color|cel_shad|comic|manga|chibi|abstract|surreal|\b3d\b|cg\b|animation/,
  },
];

const FALLBACK = "other";

export const SEMANTIC_GROUP_NAMES: string[] = [
  ...DEFS.map((d) => d.name),
  FALLBACK,
];

/**
 * Returns the semantic sub-group name for a tag.
 * Tests tags in definition order; first match wins.
 * Intended for use with General and Unknown category tags.
 */
export function getSemanticGroup(tag: string): string {
  const lower = tag.toLowerCase().replace(/_/g, " ");
  // Also test with underscores for compound patterns
  const under = tag.toLowerCase();
  for (const def of DEFS) {
    if (def.pattern.test(lower) || def.pattern.test(under)) return def.name;
  }
  return FALLBACK;
}
