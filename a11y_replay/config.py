"""Central configuration: thresholds, time windows, escalation settings.

Tune dev-split values here and freeze them before running the test split.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- segmentation ---
    analysis_width: int = 640            # downscale frames to this width before differencing
    blur_ksize: int = 5                  # Gaussian blur (odd) to suppress compression noise
    pixel_diff_threshold: int = 25       # 0-255 grayscale difference that counts as "changed"
    min_change_area_frac: float = 0.001  # ignore changes smaller than this fraction of the frame
    merge_gap_s: float = 0.25            # changed frames closer than this form one episode

    # --- clock sync (video vs key log) ---
    sync_pre_s: float = 0.4          # a key's visual effect may appear this long BEFORE its log time
    sync_post_s: float = 0.6         # ... or this long after
    sync_min_pairs: int = 3          # need at least this many key/effect pairs to trust an estimate
    sync_max_spread_s: float = 0.15  # median absolute deviation above this -> estimate not trusted

    # --- VLM escalation ---
    vlm_model: str = "claude-sonnet-5-5"
    vlm_effort: str = "medium"        # starting value; the docs advise re-running an effort sweep
    vlm_crop_pad_px: int = 60         # context around the changed region sent to the model
    vlm_max_side_px: int = 900        # downscale crops larger than this (image tokens cost money)

    # A calibration flash is a small square (about 120 x 120 points, roughly 1.5-3% of a typical frame). A change bigger
    # than this share of the frame is NOT counted as a flash, however close it is to an F9 time.
    marker_max_frac: float = 0.06

    # --- blank (black) screen: the recorded area has nothing in it (window left the area, Space switch, locked screen) ---
    blank_mean_luma: float = 8.0         # mean gray level (0-255) below this ...
    blank_p995_luma: float = 40.0        # ... and 99.5th percentile below this means "black"
    blank_min_stretch_s: float = 1.0     # only stretches at least this long are reported
    blank_warn_frac: float = 0.10        # warn if more than this share of the recording is black

    # --- benchmark calibration cue ---
    # Test apps flash a marker when this key is pressed, so clock sync works even when the
    # bug under test produces no visible effect. Detectors must ignore these key presses.
    calibration_key: str = "F9"

    # --- timeline queries ---
    effect_before_s: float = 0.15    # when asking "what did this key cause?", look this far back
    effect_after_s: float = 0.6      # ... and this far forward

    # --- key-driven focus detection (WCAG 2.4.7) ---
    # Thresholds below are UNTUNED starting values; tune on a dev split before reporting results.
    focus_keys: tuple = ("Tab",)          # key presses that move focus
    focus_before_s: float = 0.2           # "before" frame is taken this long before the key press
    focus_window_s: float = 0.3           # "after" frame is taken this long after it
    focus_min_change_area_frac: float = 0.0002  # smaller changes count as "nothing happened"
    focus_ring_hole_frac: float = 0.30    # ring-like: interior hole is at least this share of the outline
    focus_min_contrast: float = 1.5       # luminance contrast ratio (before vs after) for a clear indicator
    focus_direction_delta: float = 0.05   # luminance difference that decides appeared vs disappeared
    focus_min_component_px: int = 30      # ignore changed regions smaller than this (noise)

    # --- announcement matching (WCAG 4.1.3: seen vs heard) --- untuned starting values
    announce_window_s: float = 3.0   # speech must START within this long after the change
    announce_pre_s: float = 0.2      # ... or at most this long before it (clock slop)
    text_match_ratio: float = 80.0   # 0-100 similarity between on-screen and spoken text
    announce_min_chars: int = 6      # OCR text shorter than this does not count as a message
    announce_ignore_start_s: float = 1.0   # ignore page-load changes at the start of a recording
    announce_ocr_scale: int = 2      # upscale crops before OCR
    # Reject a similar-sounding utterance whose NUMBERS or NEGATION differ from the on-screen text
    # ("13 results found" or "No results found" for "3 results found"). Character similarity cannot see this.
    announce_content_guards: bool = True
    # ...and an announcement must SPEAK the message's content words, not just a fragment of them
    # ("Upload failed" does not announce "Upload failed: file too large"). Such partial matches become
    # needs_review, never a pass and never a finding. Word match: 0-100 similarity per word.
    announce_min_coverage: float = 0.6
    announce_word_match: float = 75.0

    # --- flash detection (WCAG 2.3.1) ---
    # Approximation of the GENERAL flash definition only. Red flash is NOT implemented.
    # Not a certified compliance test.
    flash_max_per_second: int = 3         # more than this many flashes in any 1 s violates
    flash_analysis_width: int = 160       # downscale frames to this width for luminance analysis
    flash_luminance_delta: float = 0.10   # a swing of >= 10% of max relative luminance (1.0)
    flash_dark_max: float = 0.80          # the darker image must be below this relative luminance
    # Area limit. WCAG: flashing area must exceed 25% of any 10-degree field, and a 341x256 px
    # block at 1024x768 estimates a 10-degree field. Sources disagree on whether the limit is that
    # block or 25% of it. We use the more conservative 25% reading, as a fraction of the screen.
    flash_min_area_frac: float = 0.25 * 341 * 256 / (1024 * 768)
    # A pixel counts as flashing only if the average brightness of its REGION (a cell of a grid x grid
    # division of the frame) flashes too. Without this, scrolling text (each pixel alternates between ink
    # and paper) was reported as flashing over most of the screen. Cost: a fine pattern that inverts
    # (checkerboard) has a constant regional average and is not detected.
    flash_grid: int = 4        # cells are ~6% of the frame; see docs/methodology.md for the sensitivity table

    # --- cascade ---
    escalation_confidence: float = 0.7  # below this, escalate to the VLM (tune on dev split)

    # --- evaluation ---
    time_tolerance_s: float = 1.0  # event-level match tolerance


DEFAULT = Config()