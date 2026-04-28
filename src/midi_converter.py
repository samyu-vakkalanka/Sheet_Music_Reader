"""
midi_converter.py

Converts YOLOv8 detection output for a sheet music page into a MIDI file.

Pipeline:
    1. Detect staff lines via horizontal projection profile
    2. Group staves into systems by vertical gap analysis
    3. From each system, extract first treble and first bass staff
    4. Concatenate treble staves across systems → one Part
    5. Concatenate bass staves across systems → one Part
    6. Assign pitch and rhythm to noteheads on each staff
    7. Export two-part Score to MIDI via music21
"""

import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from music21 import stream, note, chord, pitch, tempo, meter, key
from music21 import key as m21key
from pathlib import Path


# ── Class name list (must match training order) ──────────────────────────────
IN_SCOPE = [
    'noteheadBlackOnLine', 'noteheadBlackInSpace',
    'noteheadHalfOnLine', 'noteheadHalfInSpace',
    'noteheadWholeOnLine', 'noteheadWholeInSpace',
    'ledgerLine', 'stem', 'beam',
    'flag8thDown', 'flag8thUp',
    'flag16thDown', 'flag16thUp',
    'restQuarter', 'restHalf', 'restWhole', 'rest8th',
    'clefG', 'clefF',
    'timeSig4', 'timeSig3', 'timeSig2', 'timeSigCommon', 'timeSigCutCommon',
    'accidentalSharp', 'accidentalFlat', 'accidentalNatural',
    'keySharp', 'keyFlat',
    'augmentationDot'
]
ID_TO_CLASS = {i: name for i, name in enumerate(IN_SCOPE)}

# ── Music theory lookup tables ────────────────────────────────────────────────

# Number of sharps/flats → key signature string for music21
SHARP_KEYS = {
    0: 'C', 1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B', 6: 'F#', 7: 'C#'
}
FLAT_KEYS = {
    0: 'C', 1: 'F', 2: 'Bb', 3: 'Eb', 4: 'Ab', 5: 'Db', 6: 'Gb', 7: 'Cb'
}

# Order in which accidentals are added to key signatures
SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
FLAT_ORDER  = ['B', 'E', 'A', 'D', 'G', 'C', 'F']


def count_key_accidentals(all_detections, staff_lines):
    """
    Count key-signature accidentals using the RAW detection list with strict
    y-bounds (the 5 staff lines ± 5px).  Working from raw detections avoids
    the inflated list produced by assign_symbols_to_staff's 0.8x padding,
    which otherwise pulls in the adjacent staff's key symbols.
    x-bound: left of the first notehead on this specific staff.
    """
    y_min = staff_lines[0]  - 5
    y_max = staff_lines[-1] + 5

    noteheads_here = [
        d for d in all_detections
        if d['class'] in NOTEHEAD_DURATIONS
        and y_min <= (d['y0'] + d['y1']) / 2 <= y_max
    ]
    x_max = (
        min((d['x0'] + d['x1']) / 2 for d in noteheads_here)
        if noteheads_here else float('inf')
    )

    def in_key_zone(d, cls):
        cy = (d['y0'] + d['y1']) / 2
        cx = (d['x0'] + d['x1']) / 2
        return d['class'] == cls and y_min <= cy <= y_max and cx < x_max

    n_sharps = sum(1 for d in all_detections if in_key_zone(d, 'keySharp'))
    n_flats  = sum(1 for d in all_detections if in_key_zone(d, 'keyFlat'))
    return n_sharps, n_flats


def describe_key_signature(n_sharps, n_flats):
    """Return a human-readable key signature description, e.g. 'D major (2 sharps: F#, C#)'."""
    if n_sharps > 0:
        affected = ', '.join(f'{n}#' for n in SHARP_ORDER[:n_sharps])
        key_name = SHARP_KEYS.get(n_sharps, '?')
        return f'{key_name} major ({n_sharps} sharp{"s" if n_sharps > 1 else ""}: {affected})'
    elif n_flats > 0:
        affected = ', '.join(f'{n}b' for n in FLAT_ORDER[:n_flats])
        key_name = FLAT_KEYS.get(n_flats, '?')
        return f'{key_name} major ({n_flats} flat{"s" if n_flats > 1 else ""}: {affected})'
    else:
        return 'C major (no accidentals)'

# Staff position → pitch name for treble clef
# Position 0 = middle line (B4), positive = higher, negative = lower
# Positions are in half-steps from the middle line
TREBLE_POSITIONS = {
    10: ('E', 6), 9: ('D', 6), 8: ('C', 6), 7: ('B', 5), 6: ('A', 5),
    5: ('G', 5), 4: ('F', 5), 3: ('E', 5), 2: ('D', 5), 1: ('C', 5),
    0: ('B', 4),
    -1: ('A', 4), -2: ('G', 4), -3: ('F', 4), -4: ('E', 4),
    -5: ('D', 4), -6: ('C', 4), -7: ('B', 3), -8: ('A', 3),
    -9: ('G', 3), -10: ('F', 3)
}

BASS_POSITIONS = {
    10: ('G', 3), 9: ('F', 3), 8: ('E', 3), 7: ('D', 3), 6: ('C', 3),
    5: ('B', 2), 4: ('A', 2), 3: ('G', 2), 2: ('F', 2), 1: ('E', 2),
    0: ('D', 2),
    -1: ('C', 2), -2: ('B', 1), -3: ('A', 1), -4: ('G', 1),
    -5: ('F', 1), -6: ('E', 1), -7: ('D', 1), -8: ('C', 1),
    -9: ('B', 0), -10: ('A', 0)
}

# Notehead class → base duration in quarter notes
NOTEHEAD_DURATIONS = {
    'noteheadBlackOnLine':   0.25,  # will be adjusted by beams/flags
    'noteheadBlackInSpace':  0.25,
    'noteheadHalfOnLine':    2.0,
    'noteheadHalfInSpace':   2.0,
    'noteheadWholeOnLine':   4.0,
    'noteheadWholeInSpace':  4.0,
}

REST_DURATIONS = {
    'restWhole':   4.0,
    'restHalf':    2.0,
    'restQuarter': 1.0,
    'rest8th':     0.5,
}


# ── Staff line detection ──────────────────────────────────────────────────────

def detect_staff_lines(img_array, threshold_ratio=0.6, min_distance=5):
    """
    Detect staff lines using horizontal projection profile.
    Returns sorted array of y-coordinates of detected staff lines.
    """
    img_inv = 255 - img_array
    projection = np.sum(img_inv, axis=1)
    threshold = np.max(projection) * threshold_ratio
    peaks, _ = find_peaks(projection, height=threshold, distance=min_distance)
    return np.sort(peaks)


def group_lines_into_staves(peaks, lines_per_staff=5):
    """
    Group detected staff lines into staves of 5 lines each.
    Returns list of arrays, each containing 5 y-coordinates.
    """
    staves = []
    for i in range(0, len(peaks) - lines_per_staff + 1, lines_per_staff):
        group = peaks[i:i + lines_per_staff]
        if len(group) == lines_per_staff:
            staves.append(group)
    return staves


def group_staves_into_systems(staves, gap_multiplier=2.0):
    """
    Group staves into systems by detecting large vertical gaps between them.
    A gap larger than gap_multiplier * avg_staff_height indicates a new system.
    Returns list of systems, each a list of staves.
    """
    if not staves:
        return []

    # Average staff height
    staff_heights = [s[-1] - s[0] for s in staves]
    avg_height = np.mean(staff_heights)
    threshold = avg_height * gap_multiplier

    systems = [[staves[0]]]
    for i in range(1, len(staves)):
        gap = staves[i][0] - staves[i - 1][-1]
        if gap > threshold:
            systems.append([staves[i]])
        else:
            systems[-1].append(staves[i])

    return systems


# ── Symbol assignment ─────────────────────────────────────────────────────────

def assign_symbols_to_staff(detections, staff_lines):
    """
    Given detections and a staff's 5 line y-coordinates,
    return detections whose center y falls within the staff region.
    Staff region = from above top line to below bottom line with padding.
    """
    top = staff_lines[0]
    bottom = staff_lines[-1]
    staff_height = bottom - top
    padding = staff_height * 0.8  # generous padding for ledger lines

    assigned = []
    for det in detections:
        cy = (det['y0'] + det['y1']) / 2.0
        if (top - padding) <= cy <= (bottom + padding):
            assigned.append(det)
    return assigned


def get_staff_position(notehead_cy, staff_lines):
    """
    Convert notehead center y to staff position integer.
    Position 0 = middle line (index 2), positive = higher, negative = lower.
    Each step = half a space between lines.
    """
    # Space between adjacent lines
    space = (staff_lines[-1] - staff_lines[0]) / 4.0
    half_space = space / 2.0

    # Middle line y-coordinate
    middle_line_y = staff_lines[2]

    # Distance from middle line (negative = above = higher pitch)
    dist = middle_line_y - notehead_cy

    # Convert to position steps
    position = round(dist / half_space)
    return int(position)


def get_pitch_from_position(position, clef, key_sharps=0, key_flats=0):
    """
    Convert staff position to music21 pitch string.
    """
    table = TREBLE_POSITIONS if clef == 'treble' else BASS_POSITIONS

    # Clamp to table range
    position = max(min(position, 8), -10)

    if position not in table:
        position = min(table.keys(), key=lambda x: abs(x - position))

    pitch_name, octave = table[position]
    return f"{pitch_name}{octave}"


def get_duration(class_name, nearby_beams, nearby_flags):
    """
    Determine note duration in quarter note lengths.
    Black noteheads default to quarter note, adjusted by beams/flags.
    """
    base = NOTEHEAD_DURATIONS.get(class_name, 1.0)

    if 'Black' in class_name:
        # Check for flags (single notes) or beams (grouped notes)
        has_16th_flag = any('flag16th' in f for f in nearby_flags)
        has_8th_flag = any('flag8th' in f for f in nearby_flags)
        has_beam = len(nearby_beams) > 0

        if has_16th_flag:
            return 0.25  # sixteenth note
        elif has_8th_flag or has_beam:
            return 0.5   # eighth note
        else:
            return 1.0   # quarter note

    return base


# ── Core conversion ───────────────────────────────────────────────────────────

def apply_key_to_pitch(pitch_str, key_sharps, key_flats):
    """Apply key signature accidentals to a pitch name."""
    if key_sharps == 0 and key_flats == 0:
        return pitch_str
    
    if key_sharps > 0:
        k = m21key.Key(SHARP_KEYS.get(key_sharps, 'C'))
    else:
        k = m21key.Key(FLAT_KEYS.get(key_flats, 'C'))
    
    p = pitch.Pitch(pitch_str)
    
    # Compare by pitch name (letter) not pitch class
    for alteredPitch in k.alteredPitches:
        if alteredPitch.name[0] == p.name[0]:  # compare letter only e.g. 'F' == 'F'
            p.accidental = alteredPitch.accidental
            break
    
    return str(p)



def detections_to_music21_part(detections, staff_lines, clef_type,
                                key_sharps=0, key_flats=0,
                                time_sig_num=4, time_sig_den=4):
    """
    Convert a list of detections assigned to one staff into a music21 Part.
    """
    part = stream.Part()

    # Add key signature
    if key_sharps > 0:
        ks = key.Key(SHARP_KEYS.get(key_sharps, 'C'))
    elif key_flats > 0:
        ks = key.Key(FLAT_KEYS.get(key_flats, 'C'))
    else:
        ks = key.Key('C')
    part.append(ks)

    # Add time signature
    ts = meter.TimeSignature(f'{time_sig_num}/{time_sig_den}')
    part.append(ts)

    # Add tempo
    mm = tempo.MetronomeMark(number=80)
    part.append(mm)

    # Sort noteheads and rests left to right by x center
    notehead_classes = set(NOTEHEAD_DURATIONS.keys())
    rest_classes = set(REST_DURATIONS.keys())

    symbols = []
    for det in detections:
        if det['class'] in notehead_classes or det['class'] in rest_classes:
            symbols.append(det)

    symbols.sort(key=lambda d: (d['x0'] + d['x1']) / 2.0)

    # Get beams and flags for duration inference
    beams = [d for d in detections if 'beam' in d['class']]
    flags = [d for d in detections if 'flag' in d['class']]

    # Group symbols into temporal events by x-proximity.
    # Notes whose x-centers fall within chord_threshold of the current group's
    # first note are treated as simultaneous (a chord), not sequential.
    space = (staff_lines[-1] - staff_lines[0]) / 4.0
    chord_threshold = space * 1.2  # ~1 staff space — generous enough for offset noteheads

    events = []  # list of lists of symbols
    for sym in symbols:
        cx = (sym['x0'] + sym['x1']) / 2.0
        if events:
            first_cx = (events[-1][0]['x0'] + events[-1][0]['x1']) / 2.0
            if cx - first_cx < chord_threshold:
                events[-1].append(sym)
                continue
        events.append([sym])

    for group in events:
        noteheads = [s for s in group if s['class'] in notehead_classes]
        rests     = [s for s in group if s['class'] in rest_classes]

        # Rests are always solo — append them individually
        for sym in rests:
            dur = REST_DURATIONS[sym['class']]
            part.append(note.Rest(quarterLength=dur))

        if not noteheads:
            continue

        # Use the first notehead's x to look up beams/flags for duration
        ref_cx = (noteheads[0]['x0'] + noteheads[0]['x1']) / 2.0
        nearby_beams = [b for b in beams
                        if abs((b['x0'] + b['x1']) / 2.0 - ref_cx) < 100]
        nearby_flags = [f['class'] for f in flags
                        if abs((f['x0'] + f['x1']) / 2.0 - ref_cx) < 100]
        dur = get_duration(noteheads[0]['class'], nearby_beams, nearby_flags)

        pitches = []
        for sym in noteheads:
            cy = (sym['y0'] + sym['y1']) / 2.0
            position = get_staff_position(cy, staff_lines)
            pitch_str = get_pitch_from_position(position, clef_type, key_sharps, key_flats)
            pitches.append(apply_key_to_pitch(pitch_str, key_sharps, key_flats))

        if len(pitches) == 1:
            part.append(note.Note(pitches[0], quarterLength=dur))
        else:
            part.append(chord.Chord(pitches, quarterLength=dur))

    return part


def parse_detections(yolo_results):
    """
    Convert YOLOv8 results object to a list of detection dicts.
    Call this after model(image_path) returns results.
    """
    detections = []
    boxes = yolo_results.boxes

    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes.xyxy[i].cpu().numpy()
        cls_id = int(boxes.cls[i].cpu().numpy())
        conf = float(boxes.conf[i].cpu().numpy())
        class_name = ID_TO_CLASS.get(cls_id, 'unknown')

        detections.append({
            'x0': float(x0), 'y0': float(y0),
            'x1': float(x1), 'y1': float(y1),
            'class': class_name,
            'conf': conf
        })

    return detections


# ── Main entry point ──────────────────────────────────────────────────────────

def convert_page_to_midi(image_path, yolo_results, output_path,
                          conf_threshold=0.3):
    """
    Full pipeline: image + YOLOv8 results → MIDI file.

    Args:
        image_path:   Path to the sheet music page image
        yolo_results: Output of model(image_path)[0]
        output_path:  Where to save the .mid file
        conf_threshold: Minimum confidence to include a detection
    """
    # Load image for staff detection
    img = np.array(Image.open(image_path).convert('L'))

    # Parse detections
    detections = parse_detections(yolo_results)
    detections = [d for d in detections if d['conf'] >= conf_threshold]
    print(f"Using {len(detections)} detections above conf={conf_threshold}")

    # Detect staff lines and group into systems
    peaks = detect_staff_lines(img)
    staves = group_lines_into_staves(peaks)
    print(f"Detected {len(staves)} staves")

    if not staves:
        print("No staves detected — cannot convert to MIDI")
        return None, ""


     # Collect treble and bass staves
    # Use clef detection where available, fall back to alternating assumption
    treble_staves = []
    bass_staves = []

    for i, staff_lines in enumerate(staves):
        staff_dets = assign_symbols_to_staff(detections, staff_lines)
        clefs = [d for d in staff_dets
                if d['class'] in ('clefG', 'clefF')]

        if clefs:
            best_clef = max(clefs, key=lambda d: d['conf'])
            is_treble = best_clef['class'] == 'clefG'
        else:
            # Fall back to alternating: even index = treble, odd = bass
            is_treble = (i % 2 == 0)

        if is_treble:
            treble_staves.append((staff_lines, staff_dets))
        else:
            bass_staves.append((staff_lines, staff_dets))

    print(f"Treble staves: {len(treble_staves)}, Bass staves: {len(bass_staves)}")

    if not treble_staves and not bass_staves:
        print("No clef detections found — cannot determine staff type")
        return None, ""

    # Detect key signature from the first staff using raw detections + strict y-bounds.
    first_staff_lines = (treble_staves[0] if treble_staves else bass_staves[0])[0]
    detected_sharps, detected_flats = count_key_accidentals(detections, first_staff_lines)
    key_info = describe_key_signature(detected_sharps, detected_flats)
    print(f"Key signature: {key_info}")

    # Build score
    score = stream.Score()

    def build_part(staff_list, clef_type):
        """Concatenate detections from multiple staves into one Part."""
        combined_part = stream.Part()
        first_sharps = 0
        first_flats = 0
        for i, (staff_lines, staff_dets) in enumerate(staff_list):
            if i == 0:
                first_sharps, first_flats = count_key_accidentals(detections, staff_lines)
                n_sharps = first_sharps
                n_flats = first_flats

            else:
                n_sharps = 0
                n_flats = 0

            # Infer time signature from first system only
            if i == 0:
                time_dets = [d for d in staff_dets
                            if d['class'].startswith('timeSig')]
                if any(d['class'] == 'timeSigCommon' for d in time_dets):
                    ts_num, ts_den = 4, 4
                elif any(d['class'] == 'timeSigCutCommon' for d in time_dets):
                    ts_num, ts_den = 2, 2
                else:
                    nums = sorted([d for d in time_dets
                                  if d['class'] != 'timeSigCommon'],
                                 key=lambda d: d['y0'])
                    if len(nums) >= 2:
                        def sig_to_num(cls):
                            return int(cls.replace('timeSig', ''))
                        ts_num = sig_to_num(nums[0]['class'])
                        ts_den = sig_to_num(nums[1]['class'])
                    else:
                        ts_num, ts_den = 4, 4  # default
            else:
                # Keep same key signature throughout — don't reset to C major
                n_sharps = first_sharps
                n_flats = first_flats
                ts_num, ts_den = 4, 4

            part = detections_to_music21_part(
                staff_dets, staff_lines, clef_type,
                key_sharps=n_sharps,
                key_flats=n_flats,
                time_sig_num=ts_num,
                time_sig_den=ts_den
            )

            # Append elements from this staff's part into combined
            for el in part.flatten().notes:
                combined_part.append(el)

        return combined_part

    if treble_staves:
        treble_part = build_part(treble_staves, 'treble')
        treble_part.partName = 'Treble'
        score.append(treble_part)

    if bass_staves:
        bass_part = build_part(bass_staves, 'bass')
        bass_part.partName = 'Bass'
        score.append(bass_part)

    # Export to MIDI
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write('midi', fp=str(output_path))
    print(f"MIDI saved to {output_path}")
    return output_path, key_info